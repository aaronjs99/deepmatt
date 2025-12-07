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


class Thruster:
    def __init__(self, noise_std=1e-4, bias=1e-5, min_impulse=1e-6):
        self.noise_std = noise_std
        self.bias = bias
        self.min_impulse = min_impulse

    def fire(self, u_command):
        """
        Apply thruster model to commanded acceleration/force.
        """
        mag = np.linalg.norm(u_command)
        if mag < self.min_impulse:
            return np.zeros(3)

        # Add noise direction and magnitude
        noise = np.random.normal(0, self.noise_std, 3)

        # Add bias
        bias_vec = np.ones(3) * self.bias

        u_actual = u_command + noise + bias_vec

        return u_actual
