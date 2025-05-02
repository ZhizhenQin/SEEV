from typing import Tuple, Optional, List
from math import sin, cos

import torch
import numpy as np

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import Scenario, ScenarioList


class ObsAvoidDI(ControlAffineSystem):

    # Number of states and controls
    N_DIMS = 4
    N_CONTROLS = 2

    # State indices
    X = 0
    Y = 1
    VX = 2
    VY = 3

    # Control indices
    U0 = 0
    U1 = 1

    # Constant parameters
    OFFSET = 1e-2
    UNSAFE_RANGE = 0.2
    SAFE_RANGE = 0.35
    MAX_POSITION = 0.4
    MAX_VELOCITY = 0.2
    MAX_ACCLERATION = 0.2
    
    def __init__(
        self,
        nominal_params: Scenario = {},
        dt: float = 0.01,
        controller_dt: Optional[float] = 0.01,
        scenarios: Optional[ScenarioList] = [{}],
        use_l1_norm: bool = False,
        apply_offset = True,
    ):
        """
        Initialize the ObsAvoidDI system following Double Integrator vehicle model.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
            dt: the timestep to use for the simulation.
            controller_dt: the timestep for the LQR discretization. Defaults to dt.
            use_l1_norm: if True, use L1 norm for safety zones; otherwise, use L2.
        raises:
            ValueError if nominal_params are not valid for this system.
        """
        super().__init__(
            nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            scenarios=scenarios,
            use_linearized_controller=False,
        )
        self.use_l1_norm = use_l1_norm
        self.P = torch.eye(self.n_dims).float()
        self.apply_offset = apply_offset

    def validate_params(self, params: Scenario) -> bool:
        """Check if a given set of parameters is valid.

        args:
            params: a dictionary giving the parameter values for the system.
        returns:
            True if parameters are valid, False otherwise.
        """
        return True
    
    @property
    def safe_range(self):
        return ObsAvoidDI.SAFE_RANGE
    
    @property
    def unsafe_range(self):
        return ObsAvoidDI.UNSAFE_RANGE
    
    @property
    def umax(self):
        return ObsAvoidDI.MAX_ACCLERATION
    
    @property
    def initial_conditions(self):
        initial_conditions = torch.stack([self.state_limits[1],
                                          self.state_limits[0]]).T
        return initial_conditions

    @property
    def n_dims(self) -> int:
        return ObsAvoidDI.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return ObsAvoidDI.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this system.
        """
        upper_limit = torch.tensor([ObsAvoidDI.MAX_POSITION, ObsAvoidDI.MAX_POSITION,
                                    ObsAvoidDI.MAX_VELOCITY, ObsAvoidDI.MAX_VELOCITY])
        lower_limit = torch.tensor([-ObsAvoidDI.MAX_POSITION, -ObsAvoidDI.MAX_POSITION,
                                    -ObsAvoidDI.MAX_VELOCITY, -ObsAvoidDI.MAX_VELOCITY])
        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control limits for this system.
        """
        upper_limit = torch.tensor([ObsAvoidDI.MAX_ACCLERATION, ObsAvoidDI.MAX_ACCLERATION])
        lower_limit = torch.tensor([-ObsAvoidDI.MAX_ACCLERATION, -ObsAvoidDI.MAX_ACCLERATION])
        return (upper_limit, lower_limit)
    
    def safe_term(self, x):
        """
            if output > 0. then safe
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : ObsAvoidDI.Y + 1].norm(dim=-1, p=order)
        return distance - ObsAvoidDI.SAFE_RANGE

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task.

        args:
            x: a tensor of points in the state space.
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : ObsAvoidDI.Y + 1].norm(dim=-1, p=order)
        safe_mask = distance >= ObsAvoidDI.SAFE_RANGE
        return safe_mask
    
    @property
    def beta(self):
        # minimal unsafe_term
        return -ObsAvoidDI.UNSAFE_RANGE
    
    def unsafe_term(self, x):
        """
            if output < 0. then unsafe
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : ObsAvoidDI.Y + 1].norm(dim=-1, p=order)
        return distance - ObsAvoidDI.UNSAFE_RANGE
    
    def l(self, x):
        return self.unsafe_term(x)
    
    def dl_dx(self, x):
        norm = x[:, : ObsAvoidDI.Y + 1].norm(dim=-1)
        zeros = torch.zeros(x.shape[0])
        return torch.stack([
            x[:, ObsAvoidDI.X] / norm,
            x[:, ObsAvoidDI.Y] / norm,
            zeros,
            zeros
        ]).T.unsqueeze(dim = 1)

    def unsafe_mask(self, x, if_unsafe_range = False):
        """Return the mask of x indicating unsafe regions for the obstacle task.

        args:
            x: a tensor of points in the state space.
        """
        order = 1 if hasattr(self, "use_l1_norm") and self.use_l1_norm else 2
        distance = x[:, : ObsAvoidDI.Y + 1].norm(dim=-1, p=order)
        if self.apply_offset and (not if_unsafe_range):
            unsafe_mask = distance < ObsAvoidDI.OFFSET
        else:
            unsafe_mask = distance <= ObsAvoidDI.UNSAFE_RANGE
        return unsafe_mask

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set.

        args:
            x: a tensor of points in the state space.
        """
        return self.safe_mask(x)

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state.
            params: a dictionary giving the parameter values for the system. If None, default to the nominal parameters used at initialization.
        returns:
            f: bs x self.n_dims x 1 tensor.
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        f = torch.zeros((batch_size, self.n_dims, 1)).type_as(x)

        f[:, ObsAvoidDI.X, 0] = x[:, ObsAvoidDI.VX]
        f[:, ObsAvoidDI.Y, 0] = x[:, ObsAvoidDI.VY]
        return f

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-dependent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state.
            params: a dictionary giving the parameter values for the system. If None, default to the nominal parameters used at initialization.
        returns:
            g: bs x self.n_dims x self.n_controls tensor.
        """
        # Extract batch size and set up a tensor for holding the result
        batch_size = x.shape[0]
        g = torch.zeros((batch_size, self.n_dims, self.n_controls)).type_as(x)

        g[:, ObsAvoidDI.VX, ObsAvoidDI.U0] = 1.0
        g[:, ObsAvoidDI.VY, ObsAvoidDI.U1] = 1.0
        return g
    
    def integral_dynamics(self, 
                          x: torch.Tensor,
                          u: torch.Tensor,
                          dt: float
                          ):
        
        integral = torch.stack([
            x[:, ObsAvoidDI.VX] * dt + u[:, ObsAvoidDI.U0] * (dt**2/2.),
            x[:, ObsAvoidDI.VY] * dt + u[:, ObsAvoidDI.U1] * (dt**2/2.),
            u[:, ObsAvoidDI.U0] * dt,
            u[:, ObsAvoidDI.U1] * dt,
        ], dim = 1)
        return integral

    def u_nominal(
        self, x: torch.Tensor, params: Optional[Scenario] = None
    ) -> torch.Tensor:

        sign = torch.sign(x[:, :2])
        sign[sign == 0] = 1
        return sign * self.control_limits[0]

class ObsAvoidDI_BRT(ObsAvoidDI):
    
    def __init__(self, guide_fn, label_threshold,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.guide_fn = guide_fn
        self.label_threshold = label_threshold
        
    def unsafe_mask(self, x):
        
        orig_unsafe_maske = super().unsafe_mask(x, if_unsafe_range = True)
        # guide_fn <-> ABRT
        guides = self.guide_fn(x) < 0. #self.label_threshold
        return guides | orig_unsafe_maske
    
    def orig_unsafe_mask(self, x, if_unsafe_range = False):
        
        return super().unsafe_mask(x, if_unsafe_range = True)
    
    def safe_mask(self, x):
        
        orig_unsafe_maske = super().unsafe_mask(x, if_unsafe_range = True)
        guides = self.guide_fn(x) > self.label_threshold
        return guides & (~orig_unsafe_maske)
        