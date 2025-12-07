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
from deepmatt.gnc.control import PDController, SlidingModeController


def run_controller_benchmark(controller_type="lqr"):
    """Run simulation with specified controller."""
    print(f"Running {controller_type.upper()} controller...")

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
    x_deputy_true = chief_initial + target_rel_state + np.random.normal(0, 1e-6, 6)

    # Initialize controller
    if controller_type == "pd":
        controller = PDController(Kp=np.eye(3) * 0.01, Kd=np.eye(3) * 0.05)
    elif controller_type == "smc":
        controller = SlidingModeController(lambd=2.0, K=0.05, epsilon=0.01)
    else:  # LQR
        Q_lqr = np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2])
        R_lqr = np.eye(3) * 1e-1

        def get_lqr_gain(x):
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

    history = {"t": [], "pos_err": [], "fuel": 0.0}

    x_ref_rel = target_rel_state

    for t in time:
        # State
        rel_state_true = x_deputy_true - x_chief_true

        # Propagate reference
        A_ref = linearized_dynamics(x_chief_true)
        x_ref_rel = x_ref_rel + (A_ref @ x_ref_rel) * dt

        # Error
        err_state = rel_state_true - x_ref_rel

        # Control
        if controller_type == "lqr":
            K = get_lqr_gain(x_chief_true)
            u = -K @ err_state
        else:
            u = controller.compute_control(err_state)

        history["fuel"] += np.linalg.norm(u) * dt

        # Propagate
        sol_c = solve_ivp(halo_dynamics, (0, dt), x_chief_true, method="RK45")
        x_chief_next = sol_c.y[:, -1]

        def dep_dyn(t, x):
            d = halo_dynamics(t, x)
            d[3:] += u
            return d

        sol_d = solve_ivp(dep_dyn, (0, dt), x_deputy_true, method="RK45")
        x_deputy_next = sol_d.y[:, -1]

        # Record
        history["t"].append(t)
        history["pos_err"].append(np.linalg.norm(err_state[:3]))

        x_chief_true = x_chief_next
        x_deputy_true = x_deputy_next

    V_star = 1024.0
    history["fuel"] *= V_star

    return history


def benchmark_all_controllers():
    """Compare all controllers."""
    controllers = ["lqr", "pd", "smc"]

    results = {}
    for ctrl in controllers:
        results[ctrl] = run_controller_benchmark(ctrl)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    colors = {"lqr": "blue", "pd": "green", "smc": "red"}
    labels = {"lqr": "LQR (Ours)", "pd": "PD Baseline", "smc": "Sliding Mode"}

    for ctrl, hist in results.items():
        axes[0].plot(
            hist["t"],
            np.array(hist["pos_err"]) * 384400,
            label=labels[ctrl],
            color=colors[ctrl],
            linewidth=2,
        )

    axes[0].set_ylabel("Position Error (km)")
    axes[0].set_title("Controller Comparison: Tracking Performance")
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale("log")

    # Bar chart of fuel
    fuels = [results[ctrl]["fuel"] for ctrl in controllers]
    axes[1].bar(
        [labels[ctrl] for ctrl in controllers],
        fuels,
        color=[colors[ctrl] for ctrl in controllers],
    )
    axes[1].set_ylabel("Total Delta-V (m/s)")
    axes[1].set_title("Fuel Efficiency Comparison")
    axes[1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("results/validation/controller_benchmark.png", dpi=150)
    print("\nSaved plot to results/validation/controller_benchmark.png")

    # Summary
    print("\n--- Controller Benchmark Summary ---")
    print(f"{'Controller':<20} | {'Delta-V (m/s)':<15} | {'Final Error (km)':<15}")
    print("-" * 55)
    for ctrl in controllers:
        final_err = results[ctrl]["pos_err"][-1] * 384400
        print(
            f"{labels[ctrl]:<20} | {results[ctrl]['fuel']:<15.2f} | {final_err:<15.2f}"
        )


if __name__ == "__main__":
    benchmark_all_controllers()
