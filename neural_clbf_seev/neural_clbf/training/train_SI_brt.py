from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import ObsAvoidSI
from neural_clbf.models.utils import construct_network
from neural_clbf.controllers.controller_utils import normalize_with_angles
from neural_clbf.controllers import Neural_IterativeCBF_Controller, NeuralBRT_Controller

SYSTEM_CLASS = {
    "ObsAvoidSI": ObsAvoidSI,
}

def construct_networks(args, system):
    
    delta_nn = construct_network(input_dim = system.n_dims,
                                 output_dim = 1,
                                 hidden_size = args.brt_hidden_size,
                                 hidden_layers = args.brt_hidden_layers,
                                 activation = "relu",
                                 final_nonlinear = "softplus")
    
    slack_nn = construct_network(input_dim = system.n_dims,
                                 output_dim = 1,
                                 hidden_size = args.slack_hidden_size,
                                 hidden_layers = args.slack_hidden_layers,
                                 activation = "relu",
                                 final_nonlinear = "softplus")
    
    V_nn = construct_network(input_dim = system.n_dims,
                             output_dim = 1,
                             hidden_size = args.cbf_hidden_size,
                             hidden_layers = args.cbf_hidden_layers,
                             activation = "relu",
#                              final_nonlinear = "clip",
#                              clip_threshold = system.beta,
                            )
    
    return delta_nn, slack_nn, V_nn

def construct_l_recursive(delta_nn, V_nn,
                          prev_l_fn,
                          system):
    def l_fn(x):
        normalized_x = normalize_with_angles(system, x)
        new_l_vals = torch.maximum(prev_l_fn(x) - delta_nn(normalized_x).view([-1]),
                            V_nn(normalized_x).view([-1]))
        vals = prev_l_fn(x)
        vals[new_l_vals < 0.] = new_l_vals[new_l_vals < 0.]
        return vals
        
    return l_fn

def optimize_ABRT(args, system, data_module,
                  delta_nn, l_fn,):
    
    brt_learner = NeuralBRT_Controller(
        system, data_module,
        delta_nn = delta_nn,
        primal_learning_rate = args.brt_lr,
        l = l_fn,
        verbose = True,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger = False,
        reload_dataloaders_every_epoch = True,
        max_epochs = args.brt_max_epochs,
        deterministic = True,
#         progress_bar_refresh_rate = 0,
    )
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(brt_learner)
    return brt_learner.delta_nn

def optimize_CBF(args, system, data_module,
                 delta_nn, slack_nn, V_nn, l_fn):
    
    cbf_learner = Neural_IterativeCBF_Controller(
        system, data_module,
        delta_nn = delta_nn,
        V_nn = V_nn,
        slack_nn = slack_nn,
        primal_learning_rate = args.cbf_lr,
        ieloss_weight = args.cbf_ie_coeff,
        slackloss_weight = args.cbf_slack_coeff,
        adv_level = 0,
        adv_start_epochs = args.cbf_adv_start,
        lbthreshold = args.cbf_lbthreshold,
        ubthreshold = args.cbf_ubthreshold,
        eps = args.cbf_eps,
        label_threshold = args.label_threshold,
        l = l_fn,
        verbose = True,
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger = False,
        reload_dataloaders_every_epoch = True,
        max_epochs = args.cbf_max_epochs,
        deterministic = True,
#         progress_bar_refresh_rate = 0,
    )
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(cbf_learner)
    return cbf_learner.V_nn

# hardcoded for ObsAvoidDI, for now
def plot(system, model,
#          condition,
         title = "",
         N = 10000,
         plot_delta = True,
         l_fn = None,
         save_path = None,
         lbthreshold = 0.):
    
    samples = system.sample_state_space(N)
#     samples[:, -len(condition):] = condition
    out = model(normalize_with_angles(system, samples)).view(-1)
    
    if l_fn is None:
        l_fn = system.l

    if plot_delta:
        scores = l_fn(samples) - out
    else:
        scores = out
    safe_mask = scores >= lbthreshold
    unsafe_mask = scores < lbthreshold

    fig, ax = plt.subplots(figsize = [6, 6])
    circle = plt.Circle((0., 0.), system.unsafe_range, facecolor='none', edgecolor='red')
    ax.add_artist(circle)
    circle = plt.Circle((0., 0.), system.safe_range, facecolor='none', edgecolor='green')
    ax.add_artist(circle)

    ax.scatter(samples[safe_mask][:, 0], samples[safe_mask][:, 1], alpha = 0.1)
    ax.scatter(samples[unsafe_mask][:, 0], samples[unsafe_mask][:, 1], alpha = 0.1)
    if title != "":
        ax.set_title(title)
    if save_path is not None:
        plt.savefig(save_path)
    
# hardcoded for ObsAvoidDI, for now
def evaluate_delta(system, l_fn, 
                   delta_nn,
                   save_folder, counter):
    
    save_subfolder = os.path.join(save_folder, "Iter_{}".format(counter))
    if not os.path.exists(save_subfolder):
        os.makedirs(save_subfolder)
        
    torch.save(delta_nn, 
               os.path.join(save_subfolder, "ABRT_delta.pt"))
    
    title = "ABRT @ Iter {}"
    save_path = os.path.join(save_subfolder,
                             "ABRT_{}.png".format(idx+1))
    plot(system, delta_nn,
         title = title,
         plot_delta = True,
         l_fn = l_fn,
         save_path = save_path
        )
        
def evaluate_V(system, l_fn, 
               V_nn,
               save_folder, counter,
               lbthreshold):
    
    save_subfolder = os.path.join(save_folder, "Iter_{}".format(counter))
    if not os.path.exists(save_subfolder):
        os.makedirs(save_subfolder)
        
    torch.save(V_nn, 
               os.path.join(save_subfolder, "CBF.pt"))
    
    title = "CBF @ Iter {}"
    save_path = os.path.join(save_subfolder,
                             "CBF.png")
    plot(system, V_nn,
         title = title,
         plot_delta = False,
         save_path = save_path,
         lbthreshold = lbthreshold,
        )

        
def main(args):
    
    # initialize the system & data loader
    assert args.system_type in SYSTEM_CLASS
    system = SYSTEM_CLASS[args.system_type]()
    data_module = EpisodicDataModule(
        system,
        system.initial_conditions,
        trajectories_per_episode = 0,
        trajectory_length = 1,
        fixed_samples = args.sample_size,
        max_points = args.sample_size,
        val_split = 0.1,
        batch_size = args.batch_size,
        quotas={
            'unsafe': 0.4,
            'safe': 0.6,
            'boundary': 0.0,
        }
    )
    data_module.prepare_data()
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    delta_nn_list, V_nn_list, counter = [], [], 0
    l_fn = system.l
    while True:
        
        # initialize the networks for the current iteration
        delta_nn, slack_nn, V_nn = construct_networks(args, system)
#         # derive new `l' (e.g. safety specification) using the trained models
#         if counter != 0:
#             l_fn = construct_l_recursive(delta_nn_list[counter - 1],
#                                          V_nn_list[counter - 1],
#                                          l_fn, system)
        
# #         # optimize ABRT
#         delta_nn = optimize_ABRT(args, system, data_module,
#                                  delta_nn, l_fn,)
#         evaluate_delta(system, l_fn,
#                        delta_nn,
#                        args.save_folder,
#                        counter + 1)
        
        # optimize CBF
        V_nn = optimize_CBF(args, system, data_module,
                            delta_nn, slack_nn, V_nn, l_fn,)
        evaluate_V(system, l_fn,
                   V_nn,
                   args.save_folder,
                   counter + 1,
                   lbthreshold = args.cbf_lbthreshold)
        
        # terminate - TODO 
        
        # move on to next iteration
#         delta_nn_list.append(delta_nn)
#         V_nn_list.append(V_nn)
#         counter += 1
        break
        
    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--random_seed", type=int, default=111)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sample_size", type=int, default=100000)
    parser.add_argument("--system_type", type=str, default="ObsAvoidSI")
    parser.add_argument("--brt_hidden_layers", type=int, default=4)
    parser.add_argument("--brt_hidden_size", type=int, default=32)
    parser.add_argument("--brt_max_epochs", type=int, default=50)
    parser.add_argument("--brt_lr", type=float, default=1e-3)
    parser.add_argument("--cbf_hidden_layers", type=int, default=2)
    parser.add_argument("--cbf_hidden_size", type=int, default=16)
    parser.add_argument("--cbf_max_epochs", type=int, default=100)
    parser.add_argument("--cbf_adv_start", type=int, default=20)
    parser.add_argument("--cbf_eps", type=float, default=0.1)
    parser.add_argument("--cbf_ubthreshold", type=float, default=1e-4)
    parser.add_argument("--cbf_lbthreshold", type=float, default=0)
    parser.add_argument("--cbf_lr", type=float, default=1e-3)
    parser.add_argument("--cbf_ie_coeff", type=float, default=10.0)
    parser.add_argument("--cbf_slack_coeff", type=float, default=10.0)
    parser.add_argument("--slack_hidden_layers", type=int, default=2)
    parser.add_argument("--slack_hidden_size", type=int, default=16)
    parser.add_argument("--label_threshold", type=float, default=0.05)
    parser.add_argument("--save_folder", type=str, default="verifiable_iterations_0415")
    args = parser.parse_args()

    # Set random seed
    seed = args.random_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed, workers=True)
    
    main(args)
