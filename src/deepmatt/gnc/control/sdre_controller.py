"""SDRE (State-Dependent Riccati Equation) controller implementation."""

import numpy as np
from scipy.linalg import solve_continuous_are
from typing import Callable

from deepmatt.gnc.control.base_controller import Controller
from deepmatt.dynamics.cr3bp import linearized_dynamics


class SDREController(Controller):
    """State-Dependent Riccati Equation (SDRE) LQR controller.

    Solves the algebraic Riccati equation at each timestep to compute
    optimal state feedback gains K(x) for the nonlinear CR3BP dynamics.

    Attributes:
        Q: State weighting matrix (6x6)
        R: Control weighting matrix (3x3)
    """

    def __init__(self, Q: np.ndarray = None, R: np.ndarray = None):
        """Initialize SDRE controller with weighting matrices.

        Args:
            Q: State cost matrix. Defaults to diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2])
            R: Control cost matrix. Defaults to 0.1 * I_3
        """
        self.Q = Q if Q is not None else np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2])
        self.R = R if R is not None else np.eye(3) * 1e-1

    def compute_control(self, state: np.ndarray, time: float) -> np.ndarray:
        """Compute SDRE control law u = -K(x) * x.

        Args:
            state: Current error state (6,) [dx, dy, dz, dvx, dvy, dvz]
            time: Current time (unused, kept for interface compatibility)

        Returns:
            Control vector u (3,)
        """
        # Linearize dynamics around current state
        A = linearized_dynamics(state)
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)  # Control acts on acceleration

        try:
            # Solve continuous-time algebraic Riccati equation
            P = solve_continuous_are(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ B.T @ P
            u = -K @ state
            return u
        except Exception:
            # Fallback to zero control if ARE solver fails
            return np.zeros(3)
