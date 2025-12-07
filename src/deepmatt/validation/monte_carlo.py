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
import concurrent.futures
from tqdm import tqdm

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics, linearized_dynamics
from deepmatt.gnc.guidance import find_natural_formation
from deepmatt.gnc.navigation import EKF
from deepmatt.gnc.actuators import Thruster


def run_single_trial(trial_id, verbose=False):
    """Run a single GNC simulation trial with random parameters."""
    np.random.seed(trial_id)  # For reproducibility

    # Randomize parameters
    chief_initial = np.array(L1_HALO_INITIAL_STATE)
    period = 3.0

    # Get natural formation direction
    natural_dirs = find_natural_formation(chief_initial, period)
    if len(natural_dirs) == 0:
        target_rel_state = np.array([1e-4, 0, 0, 0, 0, 0])
    else:
        scale = 2.6e-5
        target_rel_state = natural_dirs[0] * scale

    # Random initial condition perturbations
    ic_pert = np.random.normal(0, 5e-6, 6)
    x_chief_true = chief_initial.copy()
    x_deputy_true = chief_initial + target_rel_state + ic_pert

    # Random sensor noise level
    sensor_noise = np.random.uniform(5e-6, 2e-5)

    # Random actuator parameters
    actuator_noise = np.random.uniform(5e-6, 2e-5)
    actuator_bias = np.random.uniform(0, 2e-6)

    # Initialize navigation
    est_error = np.random.normal(0, 1e-6, 6)
    x_rel_est = target_rel_state + est_error

    P0 = np.eye(6) * 1e-10
    Q = np.eye(6) * 1e-12
    R = np.eye(3) * sensor_noise**2

    ekf = EKF(x_rel_est, P0, Q, R)
    thruster = Thruster(noise_std=actuator_noise, bias=actuator_bias)

    # Control gains
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
    steps = int(t_max / dt)
    time = np.linspace(0, t_max, steps)

    fuel = 0.0
    pos_errors = []
    est_errors = []
    x_ref_rel = target_rel_state

    for t in time:
        # Measurement
        rel_pos_true = x_deputy_true[:3] - x_chief_true[:3]
        meas_noise = np.random.normal(0, sensor_noise, 3)
        z = rel_pos_true + meas_noise

        # Update
        ekf.update(z)
        x_est = ekf.x

        # Propagate reference
        A_ref = linearized_dynamics(x_chief_true)
        x_ref_rel = x_ref_rel + (A_ref @ x_ref_rel) * dt

        # Control
        err_state = x_est - x_ref_rel
        K = get_gain(x_chief_true)
        u_cmd = -K @ err_state
        u_act = thruster.fire(u_cmd)
        fuel += np.linalg.norm(u_act) * dt

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

        # Metrics
        pos_errors.append(
            np.linalg.norm((x_deputy_true - x_chief_true)[:3] - x_ref_rel[:3])
        )
        est_errors.append(
            np.linalg.norm(x_est[:3] - (x_deputy_true - x_chief_true)[:3])
        )

        x_chief_true = x_chief_next
        x_deputy_true = x_deputy_next

    V_star = 1024.0
    results = {
        "trial_id": trial_id,
        "total_dv": fuel * V_star,
        "final_pos_error": pos_errors[-1] * 384400 * 1000,  # meters
        "final_est_error": est_errors[-1] * 384400 * 1000,  # meters
        "max_pos_error": max(pos_errors) * 384400 * 1000,
        "mean_pos_error": np.mean(pos_errors) * 384400 * 1000,
        "success": pos_errors[-1] < 0.1,  # < 38,440 km
    }

    return results


def monte_carlo_analysis(n_trials=100):
    """Run Monte Carlo analysis with multiple trials."""
    print(f"Running Monte Carlo Analysis with {n_trials} trials...")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_trial, i) for i in range(n_trials)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=n_trials):
            results.append(future.result())

    # Statistical analysis
    dvs = [r["total_dv"] for r in results]
    pos_errs = [r["final_pos_error"] for r in results]
    est_errs = [r["final_est_error"] for r in results]
    success_rate = sum([r["success"] for r in results]) / n_trials * 100

    print("\n--- Monte Carlo Results ---")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Delta-V: {np.mean(dvs):.2f} ± {np.std(dvs):.2f} m/s")
    print(f"Final Position Error: {np.mean(pos_errs):.0f} ± {np.std(pos_errs):.0f} m")
    print(f"Final Estimation Error: {np.mean(est_errs):.0f} ± {np.std(est_errs):.0f} m")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Delta-V distribution
    axes[0, 0].hist(dvs, bins=20, edgecolor="black")
    axes[0, 0].axvline(
        np.mean(dvs), color="r", linestyle="--", label=f"Mean: {np.mean(dvs):.2f}"
    )
    axes[0, 0].set_xlabel("Delta-V (m/s)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Delta-V Distribution")
    axes[0, 0].legend()

    # Position error box plot
    axes[0, 1].boxplot(pos_errs)
    axes[0, 1].set_ylabel("Final Position Error (m)")
    axes[0, 1].set_title("Position Tracking Performance")

    # Estimation error box plot
    axes[1, 0].boxplot(est_errs)
    axes[1, 0].set_ylabel("Final Estimation Error (m)")
    axes[1, 0].set_title("Navigation Performance")

    # CDF of Delta-V
    sorted_dv = np.sort(dvs)
    cdf = np.arange(1, len(sorted_dv) + 1) / len(sorted_dv)
    axes[1, 1].plot(sorted_dv, cdf)
    axes[1, 1].set_xlabel("Delta-V (m/s)")
    axes[1, 1].set_ylabel("CDF")
    axes[1, 1].set_title("Cumulative Distribution of Delta-V")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("results/validation/monte_carlo.png", dpi=150)
    print("Saved plot to results/validation/monte_carlo.png")

    return results


if __name__ == "__main__":
    monte_carlo_analysis(n_trials=100)
