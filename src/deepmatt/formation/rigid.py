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
from deepmatt.gnc.control.lqr import lqr_gain

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics, linearized_dynamics


def constant_formation_flight():
    print(
        "Simulating Robust Multi-Agent Formation (Station Keeping + Collision Avoidance)..."
    )

    # Reference Initial State (Perfect Halo)
    ref_initial = np.array(L1_HALO_INITIAL_STATE)

    # Formation Geometry (Relative to Chief)
    dist_km = 10.0
    d = dist_km / 384400.0

    offsets = {
        "Deputy 1": np.array([d, 0, 0, 0, 0, 0]),  # +X
        "Deputy 2": np.array([-d, 0, 0, 0, 0, 0]),  # -X
        "Officer 1": np.array([0, 0, d, 0, 0, 0]),  # +Z
        "Officer 2": np.array([0, 0, -d, 0, 0, 0]),  # -Z
    }
    agent_names = ["Deputy 1", "Deputy 2", "Officer 1", "Officer 2"]

    # Initial States with Perturbations
    # Chief starts slightly off-track
    chief_initial = ref_initial + np.array([1e-6, 1e-6, 1e-6, 0, 0, 0])

    y0 = []
    y0.extend(ref_initial)  # Reference (0-5)
    y0.extend(chief_initial)  # Actual Chief (6-11)

    for name in agent_names:
        # Followers start with some error relative to desired formation
        # Perturbation: random small position error
        pert = np.random.uniform(-1e-6, 1e-6, 6)
        y0.extend(chief_initial + offsets[name] + pert)

    y0.extend([0.0] * (1 + len(agent_names)))  # DV for Chief + Followers
    y0 = np.array(y0)

    # Control Parameters
    Q = np.diag([1e4, 1e4, 1e4, 1e2, 1e2, 1e2])  # State penalty
    R = np.eye(3) * 1e-1  # Control penalty

    # Collision Avoidance Parameters
    r_safe = 2.0 / 384400.0  # 2 km safety radius
    k_rep = 1e-8  # Repulsion gain

    def get_lqr_gain(x_ref):
        A = linearized_dynamics(x_ref)

        # Solve ARE: A.T*P + P*A - P*B*R^-1*B.T*P + Q = 0
        try:
            K = lqr_gain(A, Q, R)
            return K
        except:
            return np.zeros((3, 6))

    def multi_agent_dynamics(t, y):
        # 1. Reference Dynamics (Natural)
        x_ref = y[:6]
        dx_ref = halo_dynamics(t, x_ref)

        # 2. Actual Chief Dynamics (Controlled)
        x_c = y[6:12]
        dx_c_nat = halo_dynamics(t, x_c)

        # Chief Control (Station Keeping)
        # Track Reference
        K_c = get_lqr_gain(x_ref)
        u_c_sk = -K_c @ (x_c - x_ref)

        # 3. Followers Dynamics
        # Collect all positions for collision avoidance
        positions = {"Chief": x_c[:3]}
        velocities = {"Chief": x_c[3:]}  # Not used for potential but good to have

        follower_states = {}
        for i, name in enumerate(agent_names):
            idx = 12 + i * 6
            follower_states[name] = y[idx : idx + 6]
            positions[name] = y[idx : idx + 3]

        # Calculate Control for each agent
        # We need to calculate derivatives
        dy = np.zeros_like(y)
        dy[:6] = dx_ref

        # Chief Total Control
        u_c_total = u_c_sk.copy()

        # Add collision avoidance to Chief?
        # Usually Chief is the anchor, but if an Officer is crashing into it, it might dodge.
        # Let's keep Chief steady (anchor) and make followers dodge.

        dy[6:9] = x_c[3:]
        dy[9:12] = dx_c_nat[3:] + u_c_total
        dy[12 + 4 * 6] = np.linalg.norm(u_c_total)  # Chief DV

        # Followers
        for i, name in enumerate(agent_names):
            idx_state = 12 + i * 6
            idx_dv = 12 + 4 * 6 + 1 + i

            x_f = follower_states[name]
            dx_f_nat = halo_dynamics(t, x_f)

            # Target State (Rigid Formation relative to Actual Chief)
            # Note: If Chief is moving, target is moving.
            # x_f_target = x_c + offset
            # v_f_target = v_c
            # a_f_target = a_c (which is dx_c_nat[3:] + u_c_total)

            target_pos = x_c[:3] + offsets[name][:3]
            target_vel = x_c[3:]  # Rigid translation
            target_acc = dy[9:12]  # Chief's total acceleration

            # Error
            e_pos = x_f[:3] - target_pos
            e_vel = x_f[3:] - target_vel
            error_state = np.concatenate([e_pos, e_vel])

            # Feedback Control (LQR)
            # Linearize around follower (or target)
            K_f = get_lqr_gain(x_f)
            u_fb = -K_f @ error_state

            # Feedforward (Cancel natural dynamics + match target accel)
            # u_ff = a_target - a_nat
            u_ff = target_acc - dx_f_nat[3:]

            # Collision Avoidance (Potential Function)
            u_rep = np.zeros(3)
            for other_name, other_pos in positions.items():
                if name == other_name:
                    continue

                r_vec = x_f[:3] - other_pos
                dist = np.linalg.norm(r_vec)

                if dist < r_safe:
                    # Repulsive force: F = -grad U
                    # U = 0.5 * k * (1/dist - 1/r_safe)^2
                    # F = k * (1/dist - 1/r_safe) * (1/dist^2) * (r_vec / dist)
                    # Simplified: push away proportional to 1/dist^2

                    # Let's use a simple barrier: u = k * (1/d - 1/d_safe) * n
                    # direction n = r_vec / dist

                    mag = k_rep * (1.0 / dist - 1.0 / r_safe) / (dist**2)
                    u_rep += mag * (r_vec / dist)

            # Total Control
            u_total = u_fb + u_ff + u_rep

            dy[idx_state : idx_state + 3] = x_f[3:]
            dy[idx_state + 3 : idx_state + 6] = dx_f_nat[3:] + u_total
            dy[idx_dv] = np.linalg.norm(u_total)

        return dy

    print("Integrating dynamics...")
    t_span = (0, 3.0)
    t_eval = np.linspace(0, 3.0, 1000)

    sol = solve_ivp(
        multi_agent_dynamics,
        t_span,
        y0,
        method="RK45",  # RK45 is usually faster/robust enough for this
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
    )

    # Analysis
    V_star = 1024.0

    print(f"{'Agent':<15} | {'Total Delta-V (m/s)':<20}")
    print("-" * 40)

    # Chief DV
    chief_dv = sol.y[12 + 4 * 6, -1] * V_star
    print(f"{'Chief':<15} | {chief_dv:.2f}")

    # Followers DV
    for i, name in enumerate(agent_names):
        idx_dv = 12 + 4 * 6 + 1 + i
        dv = sol.y[idx_dv, -1] * V_star
        print(f"{name:<15} | {dv:.2f}")

    # Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot Reference
    rx, ry, rz = sol.y[0], sol.y[1], sol.y[2]
    ax.plot(rx, ry, rz, "k:", linewidth=1, label="Reference Halo")

    # Plot Actual Chief
    cx, cy, cz = sol.y[6], sol.y[7], sol.y[8]
    ax.plot(cx, cy, cz, "k-", linewidth=2, label="Chief (Controlled)")

    # Plot Followers
    colors = ["r", "g", "b", "m"]
    for i, name in enumerate(agent_names):
        idx = 12 + i * 6
        fx, fy, fz = sol.y[idx], sol.y[idx + 1], sol.y[idx + 2]
        ax.plot(fx, fy, fz, color=colors[i], linestyle="--", label=name)

    ax.set_title("Robust Formation Control (Station Keeping + Collision Avoidance)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/formation/robust_formation.png")
    plt.close()
    print("Saved plot to results/formation/robust_formation.png")

    # Animation
    from deepmatt.core.animator import animate_formation

    print("Generating animation...")
    chief_state = sol.y[6:12].T
    deputy_states = [
        sol.y[12 + i * 6 : 12 + (i + 1) * 6].T for i in range(len(agent_names))
    ]
    animate_formation(
        sol.t,
        chief_state,
        deputy_states,
        save_path="results/formation/formation_animation.gif",
    )

    return sol


def main():
    constant_formation_flight()


if __name__ == "__main__":
    main()
