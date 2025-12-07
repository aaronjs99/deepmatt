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
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics, linearized_dynamics
from deepmatt.gnc.guidance import find_natural_formation
from deepmatt.gnc.navigation import EKF
from deepmatt.gnc.actuators import Thruster


def run_failure_scenario(failure_type="thruster", failure_time=1.0):
    """
    Run GNC simulation with failure injection.

    Args:
        failure_type: 'thruster', 'comms', or 'sensor_degradation'
        failure_time: Time to inject failure
    """
    print(f"Running {failure_type} failure scenario...")

    # Setup
    chief_initial = np.array(L1_HALO_INITIAL_STATE)
    period = 3.0

    natural_dirs = find_natural_formation(chief_initial, period)
    scale = 2.6e-5
    target_rel_state = (
        natural_dirs[0] * scale
        if len(natural_dirs) > 0
        else np.array([1e-4, 0, 0, 0, 0, 0])
    )

    x_chief_true = chief_initial.copy()
    x_deputy_true = chief_initial + target_rel_state

    # Navigation
    x_rel_est = target_rel_state.copy()
    P0 = np.eye(6) * 1e-10
    Q = np.eye(6) * 1e-12
    R = np.eye(3) * 1e-10

    ekf = EKF(x_rel_est, P0, Q, R)
    thruster = Thruster(noise_std=1e-5, bias=1e-6)

    # Control
    Q_lqr = np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2])
    R_lqr = np.eye(3) * 1e-1

    def get_gain(x):
        A = linearized_dynamics(x)
        B = np.zeros((6, 3))
        B[3:, :] = np.eye(3)
        try:
            P = solve_continuous_are(A, B, Q_lqr, R_lqr)
            K = np.linalg.inv(R_lqr) @ B.T @ P
            return K
        except:
            return np.zeros((3, 6))

    # Simulation
    dt = 0.01
    t_max = 3.0
    time = np.linspace(0, t_max, int(t_max / dt))

    history = {"t": [], "pos_err": [], "est_err": [], "control": [], "fuel": 0.0}

    x_ref_rel = target_rel_state
    thruster_failed = False
    comms_dropout = False
    sensor_noise_factor = 1.0

    for t in time:
        # Inject failure
        if t >= failure_time:
            if failure_type == "thruster" and not thruster_failed:
                print(f"  Thruster failure at t={t:.2f}")
                thruster_failed = True
            elif failure_type == "comms" and not comms_dropout:
                print(f"  Communication dropout at t={t:.2f}")
                comms_dropout = True
            elif failure_type == "sensor_degradation":
                sensor_noise_factor = 1.0 + 10.0 * (t - failure_time)

        # Measurement
        rel_pos_true = x_deputy_true[:3] - x_chief_true[:3]

        if not comms_dropout:
            meas_noise = np.random.normal(0, 1e-5 * sensor_noise_factor, 3)
            z = rel_pos_true + meas_noise
            ekf.update(z)

        x_est = ekf.x

        # Propagate reference
        A_ref = linearized_dynamics(x_chief_true)
        x_ref_rel = x_ref_rel + (A_ref @ x_ref_rel) * dt

        # Control
        err_state = x_est - x_ref_rel
        K = get_gain(x_chief_true)
        u_cmd = -K @ err_state

        # Actuation with failure
        if thruster_failed:
            u_act = np.zeros(3)  # Stuck off
        else:
            u_act = thruster.fire(u_cmd)

        history["fuel"] += np.linalg.norm(u_act) * dt

        # Propagate
        sol_c = solve_ivp(halo_dynamics, (0, dt), x_chief_true, method="RK45")
        x_chief_next = sol_c.y[:, -1]

        def dep_dyn(t, x):
            d = halo_dynamics(t, x)
            d[3:] += u_act
            return d

        sol_d = solve_ivp(dep_dyn, (0, dt), x_deputy_true, method="RK45")
        x_deputy_next = sol_d.y[:, -1]

        ekf.predict(u_cmd, dt, x_chief_true)

        # Record
        history["t"].append(t)
        history["pos_err"].append(np.linalg.norm(rel_pos_true[:3] - x_ref_rel[:3]))
        history["est_err"].append(np.linalg.norm(x_est[:3] - rel_pos_true[:3]))
        history["control"].append(np.linalg.norm(u_act))

        x_chief_true = x_chief_next
        x_deputy_true = x_deputy_next

    return history


def analyze_all_failures():
    """Run and plot all failure scenarios."""
    scenarios = ["thruster", "comms", "sensor_degradation"]

    results = {}
    for scenario in scenarios:
        results[scenario] = run_failure_scenario(scenario, failure_time=1.0)

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    colors = {"thruster": "red", "comms": "orange", "sensor_degradation": "purple"}

    for scenario, hist in results.items():
        axes[0].plot(
            hist["t"],
            np.array(hist["pos_err"]) * 384400,
            label=scenario.replace("_", " ").title(),
            color=colors[scenario],
        )
        axes[1].plot(
            hist["t"], np.array(hist["est_err"]) * 384400 * 1000, color=colors[scenario]
        )
        axes[2].plot(hist["t"], hist["control"], color=colors[scenario])

    axes[0].axvline(
        1.0, color="k", linestyle="--", alpha=0.3, label="Failure Injection"
    )
    axes[0].set_ylabel("Position Error (km)")
    axes[0].set_title("Formation Keeping Under Failures")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].axvline(1.0, color="k", linestyle="--", alpha=0.3)
    axes[1].set_ylabel("Estimation Error (m)")
    axes[1].set_title("Navigation Performance Under Failures")
    axes[1].grid(True)

    axes[2].axvline(1.0, color="k", linestyle="--", alpha=0.3)
    axes[2].set_ylabel("Control Effort")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Control Activity")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("results/validation/failure_modes.png", dpi=150)
    print("\nSaved plot to results/validation/failure_modes.png")

    # Summary
    print("\n--- Failure Mode Analysis Summary ---")
    for scenario, hist in results.items():
        final_err = hist["pos_err"][-1] * 384400
        print(f"{scenario.replace('_', ' ').title()}: Final Error = {final_err:.2f} km")


if __name__ == "__main__":
    analyze_all_failures()
