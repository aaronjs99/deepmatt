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
Gymnasium Environment for Formation Flying Control.

This environment simulates relative motion of a deputy spacecraft
near a chief spacecraft in a Halo orbit using CR3BP dynamics.

Features:
- RK4 integration for accurate dynamics
- Configurable reward shaping
- Proper termination conditions
- Observation normalization
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from deepmatt.dynamics.cr3bp import cr3bp_dynamics
from deepmatt.core.params import mu, L_star, T_star


class DeepMAttFormationEnv(gym.Env):
    """
    Formation Flying Environment for Cislunar Space.

    Observation Space: [x, y, z, vx, vy, vz] relative state (6D)
    Action Space: [ux, uy, uz] thrust commands normalized to [-1, 1]

    Reward Design (matches paper Eq. 12):
        r = -alpha * ||pos_error|| - beta * ||vel_error|| - lambda * ||u|| + bonus

    Termination:
        - Position error > max_error (divergence)
        - Truncated after max_steps
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        chief_history,
        max_steps=1000,
        dt=0.01,
        max_thrust=1e-4,
        max_error=0.01,
        alpha=1.0,
        beta=0.1,
        lambd=0.5,
        bonus=10.0,
        bonus_threshold=1e-5,
        use_rk4=True,
        normalize_obs=True,
    ):
        """
        Initialize the formation flying environment.

        Args:
            chief_history: (N, 6) array of chief state trajectory
            max_steps: Maximum steps per episode
            dt: Integration timestep (non-dimensional)
            max_thrust: Maximum thrust acceleration (non-dimensional)
            max_error: Position error threshold for termination
            alpha: Position error weight in reward
            beta: Velocity error weight in reward
            lambd: Control effort weight in reward
            bonus: Bonus for maintaining formation
            bonus_threshold: Position error threshold for bonus
            use_rk4: Use RK4 integration (True) or Euler (False)
            normalize_obs: Normalize observations
        """
        super(DeepMAttFormationEnv, self).__init__()

        # Store chief trajectory
        self.chief_history = np.asarray(chief_history, dtype=np.float64)
        if self.chief_history.ndim != 2 or self.chief_history.shape[1] != 6:
            raise ValueError("chief_history must be shape (N, 6)")

        # Environment parameters
        self.max_steps = max_steps
        self.dt = dt
        self.max_thrust = max_thrust
        self.max_error = max_error
        self.use_rk4 = use_rk4
        self.normalize_obs = normalize_obs

        # Reward parameters (paper Eq. 12)
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.bonus = bonus
        self.bonus_threshold = bonus_threshold

        # Action space: thrust commands normalized to [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: relative state
        # Bounds based on expected operational range
        if normalize_obs:
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0, shape=(6,), dtype=np.float32
            )
            # Normalization scales (approx 100 km position, 10 m/s velocity)
            self.pos_scale = 1e-4  # ~38 km in non-dim
            self.vel_scale = 1e-5  # ~3.8 m/s in non-dim
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )
            self.pos_scale = 1.0
            self.vel_scale = 1.0

        # State
        self.state = None
        self.chief_idx = 0
        self.current_step = 0
        self.cumulative_fuel = 0.0

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Random initial offset (position ~1km, velocity ~1cm/s)
        if self.np_random is not None:
            pos_err = self.np_random.normal(0, 1e-5, 3)
            vel_err = self.np_random.normal(0, 1e-6, 3)
        else:
            pos_err = np.random.normal(0, 1e-5, 3)
            vel_err = np.random.normal(0, 1e-6, 3)

        self.state = np.concatenate([pos_err, vel_err])

        # Reset counters
        if options and "start_idx" in options:
            self.chief_idx = options["start_idx"] % len(self.chief_history)
        else:
            self.chief_idx = 0

        self.current_step = 0
        self.cumulative_fuel = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one environment step."""
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        u_cmd = action * self.max_thrust

        # Get current chief state
        xc = self.chief_history[self.chief_idx]
        xd = xc + self.state

        # Integrate dynamics
        if self.use_rk4:
            new_state = self._rk4_step(xc, xd, u_cmd)
        else:
            new_state = self._euler_step(xc, xd, u_cmd)

        self.state = new_state

        # Update counters
        self.current_step += 1
        self.chief_idx = (self.chief_idx + 1) % len(self.chief_history)
        self.cumulative_fuel += np.linalg.norm(u_cmd)

        # Compute reward
        pos_error = np.linalg.norm(self.state[:3])
        vel_error = np.linalg.norm(self.state[3:])
        fuel_cost = np.linalg.norm(u_cmd)

        reward = -(
            self.alpha * pos_error + self.beta * vel_error + self.lambd * fuel_cost
        )

        # Bonus for maintaining tight formation
        if pos_error < self.bonus_threshold:
            reward += self.bonus

        # Check termination
        terminated = pos_error > self.max_error
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _euler_step(self, xc, xd, u_cmd):
        """Simple Euler integration."""
        dxd = np.array(cr3bp_dynamics(0, xd))
        dxc = np.array(cr3bp_dynamics(0, xc))

        dxd[3:] += u_cmd  # Add control to deputy
        d_rel = dxd - dxc

        return self.state + d_rel * self.dt

    def _rk4_step(self, xc, xd, u_cmd):
        """4th-order Runge-Kutta integration."""
        dt = self.dt

        def relative_dynamics(rel_state, xc_local):
            xd_local = xc_local + rel_state
            dxd = np.array(cr3bp_dynamics(0, xd_local))
            dxc = np.array(cr3bp_dynamics(0, xc_local))
            dxd[3:] += u_cmd
            return dxd - dxc

        # RK4 with chief state interpolation (simplified: use same xc)
        k1 = relative_dynamics(self.state, xc)
        k2 = relative_dynamics(self.state + 0.5 * dt * k1, xc)
        k3 = relative_dynamics(self.state + 0.5 * dt * k2, xc)
        k4 = relative_dynamics(self.state + dt * k3, xc)

        return self.state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _get_obs(self):
        """Get normalized observation."""
        obs = self.state.copy()
        if self.normalize_obs:
            obs[:3] /= self.pos_scale
            obs[3:] /= self.vel_scale
        return obs.astype(np.float32)

    def _get_info(self):
        """Get additional info dict."""
        return {
            "pos_error": np.linalg.norm(self.state[:3]),
            "vel_error": np.linalg.norm(self.state[3:]),
            "cumulative_fuel": self.cumulative_fuel,
            "chief_idx": self.chief_idx,
            "step": self.current_step,
        }

    def render(self):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Clean up resources."""
        pass
