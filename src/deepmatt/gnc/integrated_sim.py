"""Reproducible sampled GNC simulation, independent of plotting and storage.

Copyright (c) 2025 University of California, Los Angeles.
Author: Aaron John-Sabu.

Guidance follows an unforced deputy trajectory initialized with a configured
offset. This does not certify that the reference is periodic or bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

from deepmatt.dynamics.cr3bp import cr3bp_dynamics, linearized_dynamics
from deepmatt.gnc.actuators import Thruster
from deepmatt.gnc.config import GNCConfig
from deepmatt.gnc.control.lqr import lqr_gain
from deepmatt.gnc.navigation import EKF


@dataclass(frozen=True)
class GNCResult:
    """States include both endpoints; controls apply over successive intervals.

    Every array uses normalized rotating-frame CR3BP units. Absolute states
    are [x, y, z, vx, vy, vz]. The navigation estimate is deputy minus chief.
    """

    config: GNCConfig
    time: np.ndarray
    chief_state: np.ndarray
    reference_state: np.ndarray
    deputy_state: np.ndarray
    estimated_relative_state: np.ndarray
    covariance: np.ndarray
    control_requested: np.ndarray
    control_command: np.ndarray
    control_applied: np.ndarray
    delta_v: np.ndarray

    @property
    def position_error(self):
        return np.linalg.norm(
            self.deputy_state[:, :3] - self.reference_state[:, :3], axis=1
        )

    @property
    def velocity_error(self):
        return np.linalg.norm(
            self.deputy_state[:, 3:] - self.reference_state[:, 3:], axis=1
        )

    @property
    def estimation_position_error(self):
        truth = self.deputy_state[:, :3] - self.chief_state[:, :3]
        return np.linalg.norm(self.estimated_relative_state[:, :3] - truth, axis=1)

    def summary(self):
        """Separate position/velocity metrics; physical values require scales."""
        system = self.config.system
        length = (
            None if system.length_scale_km is None else system.length_scale_km * 1000
        )
        velocity = (
            None
            if length is None or system.time_scale_s is None
            else length / system.time_scale_s
        )
        position_error = self.position_error
        return {
            "schema_version": 1,
            "status": "completed",
            "model": "normalized rotating-frame CR3BP; sampled LQR; relative-position EKF",
            "reference": "unforced deputy initialized with relative_offset; no periodicity guarantee",
            "system": system.name,
            "seed": self.config.seed,
            "duration_nd": float(self.time[-1]),
            "sample_count": len(self.time),
            "interval_count": len(self.control_applied),
            "final_position_error_nd": float(position_error[-1]),
            "rms_position_error_nd": float(np.sqrt(np.mean(position_error**2))),
            "max_position_error_nd": float(np.max(position_error)),
            "final_velocity_error_nd": float(self.velocity_error[-1]),
            "final_estimation_position_error_nd": float(
                self.estimation_position_error[-1]
            ),
            "total_delta_v_nd": float(self.delta_v[-1]),
            "final_position_error_m": None
            if length is None
            else float(position_error[-1] * length),
            "final_estimation_position_error_m": None
            if length is None
            else float(self.estimation_position_error[-1] * length),
            "final_velocity_error_m_s": None
            if velocity is None
            else float(self.velocity_error[-1] * velocity),
            "total_delta_v_m_s": None
            if velocity is None
            else float(self.delta_v[-1] * velocity),
            "saturated_interval_count": int(
                np.count_nonzero(
                    np.linalg.norm(self.control_requested, axis=1)
                    > self.config.max_acceleration
                )
            ),
        }


def _time_grid(config):
    full_steps = math.floor(config.duration / config.time_step)
    time = np.arange(full_steps + 1, dtype=float) * config.time_step
    if full_steps and math.isclose(time[-1], config.duration, rel_tol=1e-12):
        time[-1] = config.duration
    else:
        time = np.r_[time, config.duration]
    return time


def run_gnc_simulation(config: GNCConfig | None = None) -> GNCResult:
    """Run local simulation only; fail explicitly on incomplete integrations.

    Noise is drawn only at sample boundaries with a run-local Generator.
    A chief, unforced reference, and controlled deputy share one ODE solve
    per interval. The estimator knows commanded acceleration, not actuator
    noise or the deputy's true velocity.
    """
    config = GNCConfig() if config is None else config
    if not isinstance(config, GNCConfig):
        raise ValueError("config must be a GNCConfig.")
    rng = np.random.default_rng(config.seed)
    time = _time_grid(config)
    samples = len(time)
    chief = np.empty((samples, 6))
    reference = np.empty_like(chief)
    deputy = np.empty_like(chief)
    estimate = np.empty_like(chief)
    covariance = np.empty((samples, 6, 6))
    requested = np.empty((samples - 1, 3))
    command = np.empty_like(requested)
    applied = np.empty_like(requested)
    delta_v = np.zeros(samples)

    chief[0] = config.chief_initial_state
    reference[0] = chief[0] + config.relative_offset
    deputy[0] = reference[0] + config.initial_relative_error
    ekf = EKF(
        deputy[0] - chief[0] + rng.normal(0, config.initial_estimate_std, 6),
        np.eye(6) * config.initial_estimate_std**2,
        np.eye(6) * config.process_noise_variance,
        np.eye(3) * config.measurement_std**2,
        system=config.system,
    )
    thruster = Thruster(
        noise_std=config.thruster_noise_std,
        bias=config.thruster_bias,
        min_impulse=config.thruster_deadband,
        rng=rng,
        max_acceleration=config.max_acceleration,
    )
    state_cost = np.diag(config.lqr_state_weights)
    input_cost = np.eye(3) * config.lqr_control_weight

    def observe(index):
        measurement = deputy[index, :3] - chief[index, :3]
        measurement = measurement + rng.normal(0, config.measurement_std, 3)
        ekf.update(measurement)
        if not np.all(np.isfinite(ekf.x)) or not np.all(np.isfinite(ekf.P)):
            raise RuntimeError(f"Navigation became non-finite at t={time[index]:.8g}.")
        estimate[index] = ekf.x
        covariance[index] = ekf.P

    observe(0)
    for index, step in enumerate(np.diff(time)):
        start, end = time[index : index + 2]
        try:
            jacobian = linearized_dynamics(chief[index], system=config.system)
            gain = lqr_gain(jacobian, state_cost, input_cost)
            error = estimate[index] - (reference[index] - chief[index])
            requested[index] = -gain @ error
            if not np.all(np.isfinite(requested[index])):
                raise RuntimeError("LQR produced a non-finite command.")
            magnitude = np.linalg.norm(requested[index])
            command[index] = (
                requested[index] * min(1.0, config.max_acceleration / magnitude)
                if magnitude
                else 0
            )
            if np.linalg.norm(command[index]) <= config.thruster_deadband:
                command[index] = 0
            applied[index] = thruster.fire(command[index])

            def dynamics(t, state):
                derivative = np.concatenate(
                    [
                        cr3bp_dynamics(t, state[part : part + 6], system=config.system)
                        for part in (0, 6, 12)
                    ]
                )
                derivative[15:18] += applied[index]
                return derivative

            initial = np.concatenate([chief[index], reference[index], deputy[index]])
            solution = solve_ivp(
                dynamics,
                (start, end),
                initial,
                method="DOP853",
                t_eval=[end],
                rtol=config.rtol,
                atol=config.atol,
            )
            if not solution.success:
                raise RuntimeError(f"Physics integration failed: {solution.message}")
            if solution.y.shape != (18, 1) or not np.all(np.isfinite(solution.y)):
                raise RuntimeError(
                    "Physics integration returned incomplete or non-finite states."
                )
            chief[index + 1], reference[index + 1], deputy[index + 1] = solution.y[
                :, -1
            ].reshape(3, 6)
            ekf.predict(command[index], step, chief[index])
            observe(index + 1)
            delta_v[index + 1] = delta_v[index] + np.linalg.norm(applied[index]) * step
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            raise RuntimeError(
                f"GNC failed over t={start:.8g} to {end:.8g}: {exc}"
            ) from exc

    return GNCResult(
        config,
        time,
        chief,
        reference,
        deputy,
        estimate,
        covariance,
        requested,
        command,
        applied,
        delta_v,
    )


def integrated_gnc_sim(config=None, *, output_dir="results/runs", plot=True):
    """Backward-compatible executable entry point, with inspectable outputs."""
    from deepmatt.output.gnc import save_gnc_result

    result = run_gnc_simulation(config)
    destination = save_gnc_result(result, output_dir, plot=plot)
    print(f"GNC simulation completed. Results: {destination}")
    return result


if __name__ == "__main__":
    integrated_gnc_sim()
