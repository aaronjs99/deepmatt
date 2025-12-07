# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John-Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

import numpy as np
from deepmatt.gnc.control.base_controller import Controller


class SlidingModeController(Controller):
    def __init__(self, lambd, K, epsilon=0.01):
        """
        Sliding Mode Controller.

        Args:
            lambd: Sliding surface slope (scalar or 3,)
            K: Switching gain (scalar or 3,)
            epsilon: Boundary layer thickness
        """
        self.lambd = lambd
        self.K = K
        self.epsilon = epsilon

    def compute_control(self, state: np.ndarray, time: float = 0.0) -> np.ndarray:
        """Compute control input.

        Args:
            state: Relative state error [dx, dy, dz, dvx, dvy, dvz]
            time: Current time (unused)

        Returns:
            u: Control acceleration (3,)
        """
        e_pos = state[:3]
        e_vel = state[3:]

        # Sliding surface: s = e_vel + lambda * e_pos
        s = e_vel + self.lambd * e_pos

        # Sat function (smooth approximation of sign)
        def sat(x, eps):
            return np.clip(x / eps, -1, 1)

        # Control: u = -K * sat(s)
        u = -self.K * sat(s, self.epsilon)

        return u
