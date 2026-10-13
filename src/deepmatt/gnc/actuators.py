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
    def __init__(
        self,
        noise_std=1e-4,
        bias=1e-5,
        min_impulse=1e-6,
        *,
        rng=None,
        max_acceleration=None,
    ):
        """Acceleration deadband, additive noise/bias, then an optional norm cap.

        ``min_impulse`` is the legacy name for a command acceleration deadband.
        Inject a Generator for isolated randomness; omitted RNG retains legacy
        global NumPy seeding behavior for existing experiments.
        """
        for name, value in (
            ("noise_std", noise_std),
            ("bias", bias),
            ("min_impulse", min_impulse),
        ):
            if not np.isfinite(value) or (name != "bias" and value < 0):
                raise ValueError(
                    f"{name} must be finite and nonnegative (bias may be signed)."
                )
        if max_acceleration is not None and (
            not np.isfinite(max_acceleration) or max_acceleration <= 0
        ):
            raise ValueError("max_acceleration must be finite and positive.")
        self.noise_std = noise_std
        self.bias = bias
        self.min_impulse = min_impulse
        self.rng = rng
        self.max_acceleration = max_acceleration

    def fire(self, u_command):
        """
        Apply thruster model to commanded acceleration/force.
        """
        u_command = np.asarray(u_command, dtype=float)
        if u_command.shape != (3,) or not np.all(np.isfinite(u_command)):
            raise ValueError(
                "Thruster command must contain three finite accelerations."
            )
        mag = np.linalg.norm(u_command)
        if mag <= self.min_impulse:
            return np.zeros(3)

        # Add noise direction and magnitude
        noise = (np.random if self.rng is None else self.rng).normal(
            0, self.noise_std, 3
        )

        # Add bias
        bias_vec = np.ones(3) * self.bias

        u_actual = u_command + noise + bias_vec
        actual_magnitude = np.linalg.norm(u_actual)
        if (
            self.max_acceleration is not None
            and actual_magnitude > self.max_acceleration
        ):
            u_actual = u_actual * (self.max_acceleration / actual_magnitude)

        return u_actual
