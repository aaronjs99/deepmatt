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

"""
Extended Kalman Filter (EKF) for Relative Navigation.

This module implements an EKF for estimating relative position and velocity
between chief and deputy spacecraft in formation flying scenarios.

Classes:
    EKF: Extended Kalman Filter with predict/update steps

Reference:
    Paper Section IV.B: Navigation Architecture
"""

import numpy as np
from scipy.linalg import expm
from deepmatt.dynamics.cr3bp import linearized_dynamics
from deepmatt.core.system import CR3BPSystem, EARTH_MOON_SYSTEM
from deepmatt.gnc.control.lqr import acceleration_input_matrix


class EKF:
    def __init__(self, x0, P0, Q, R, *, system: CR3BPSystem = EARTH_MOON_SYSTEM):
        """
        Extended Kalman Filter for Relative Navigation.
        State: Relative Position and Velocity (6x1)
        """
        self.system = system
        self.x = np.array(x0, dtype=float, copy=True)
        self.P = np.array(P0, dtype=float, copy=True)
        self.Q = np.array(Q, dtype=float, copy=True)
        self.R = np.array(R, dtype=float, copy=True)
        for name, shape in (("x", (6,)), ("P", (6, 6)), ("Q", (6, 6)), ("R", (3, 3))):
            value = getattr(self, name)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite array of shape {shape}.")

    def predict(self, u, dt, chief_state):
        """
        Propagate state and covariance.
        Model: x_k+1 = Phi * x_k + B * u
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Navigation time step must be finite and positive.")
        u = np.asarray(u, dtype=float)
        if u.shape != (3,) or not np.all(np.isfinite(u)):
            raise ValueError(
                "Navigation control must contain three finite accelerations."
            )
        A = linearized_dynamics(chief_state, system=self.system)
        # Exact zero-order-hold discretization of the frozen local model.
        augmented = np.zeros((9, 9))
        augmented[:6, :6] = A
        augmented[:6, 6:] = acceleration_input_matrix()
        transition = expm(augmented * dt)
        Phi = transition[:6, :6]
        self.x = Phi @ self.x + transition[:6, 6:] @ u
        # Q retains the existing contract: covariance added per predict call.
        self.P = Phi @ self.P @ Phi.T + self.Q
        self.P = (self.P + self.P.T) * 0.5

    def update(self, z):
        """
        Update with measurement.
        Measurement: Relative Position (3x1) (assuming GPS/Link gives pos directly for simplicity, or Range/Bearing converted)
        Let's assume we measure [x, y, z] directly with noise.
        H = [I 0]
        """
        z = np.asarray(z, dtype=float)
        if z.shape != (3,) or not np.all(np.isfinite(z)):
            raise ValueError(
                "Navigation measurement must contain three finite positions."
            )
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)

        # Innovation
        y = z - H @ self.x

        # Innovation Covariance
        S = H @ self.P @ H.T + self.R

        # Kalman Gain
        K = np.linalg.solve(S, H @ self.P).T

        # Update State
        self.x = self.x + K @ y

        # Update Covariance
        residual = np.eye(6) - K @ H
        self.P = residual @ self.P @ residual.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) * 0.5

        return self.x
