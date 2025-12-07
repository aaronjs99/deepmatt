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


class PDController(Controller):
    """Proportional-Derivative Controller."""

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray):
        """Initialize PD Controller.

        Args:
            Kp: Proportional gain matrix (3x3)
            Kd: Derivative gain matrix (3x3)
        """
        self.Kp = Kp
        self.Kd = Kd

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

        u = -self.Kp @ e_pos - self.Kd @ e_vel
        return u
