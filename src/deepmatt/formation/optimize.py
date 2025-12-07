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
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import cr3bp_dynamics as halo_dynamics


def optimize_formation():
    print("Optimizing Formation Orientation...")

    # Chief initial state
    chief_initial = np.array(L1_HALO_INITIAL_STATE)

    # Base offsets (Cross configuration)
    dist_km = 10.0
    d = dist_km / 384400.0

    # Base positions relative to Chief (before rotation)
    # [x, y, z]
    base_positions = [
        np.array([d, 0, 0]),  # Dep 1
        np.array([-d, 0, 0]),  # Dep 2
        np.array([0, 0, d]),  # Off 1
        np.array([0, 0, -d]),  # Off 2
    ]

    # Time span for evaluation (shorter for speed during optimization?)
    # Let's use full 3.0 units to be accurate.
    t_span = (0, 3.0)
    t_eval = np.linspace(0, 3.0, 100)  # Fewer points for speed

    # Pre-compute Chief trajectory to avoid re-integrating it every time?
    # No, solve_ivp overhead is small compared to integration.
    # But we can integrate Chief once and interpolate.
    # For simplicity, let's just integrate everything.

    def cost_function(angles):
        # angles: [roll, pitch, yaw] in radians
        r = Rotation.from_euler("xyz", angles)
        R = r.as_matrix()

        # Rotated offsets
        rotated_offsets = []
        for pos in base_positions:
            rot_pos = R @ pos
            # Velocity offset is 0 for rigid formation
            offset = np.concatenate([rot_pos, [0, 0, 0]])
            rotated_offsets.append(offset)

        # Run simulation
        # We can use a simplified integrator or just the same one
        # To speed up, we can use lower tolerance for optimization

        y0 = []
        y0.extend(chief_initial)
        for offset in rotated_offsets:
            y0.extend(chief_initial + offset)
        y0.extend([0.0] * 4)  # DV
        y0 = np.array(y0)

        def dynamics(t, y):
            xc = y[:6]
            dxc = halo_dynamics(t, xc)

            dy = [0.0] * len(y)
            dy[:6] = dxc

            for i in range(4):
                idx_state = 6 + i * 6
                idx_dv = 30 + i
                xf = y[idx_state : idx_state + 6]

                dxf_nat = halo_dynamics(t, xf)
                af_nat = np.array(dxf_nat[3:])
                at_target = np.array(dxc[3:])

                u = at_target - af_nat
                u_mag = np.sqrt(np.sum(u**2))

                dy[idx_state : idx_state + 3] = dxc[:3]
                dy[idx_state + 3 : idx_state + 6] = dxc[3:]
                dy[idx_dv] = u_mag
            return dy

        sol = solve_ivp(dynamics, t_span, y0, method="RK45", rtol=1e-6, atol=1e-9)

        total_dv = np.sum(sol.y[30:, -1])
        return total_dv

    # Initial guess: 0, 0, 0
    x0 = [0.0, 0.0, 0.0]

    print("Starting optimization (this may take a minute)...")
    res = minimize(cost_function, x0, method="Nelder-Mead", tol=1e-4)

    print("\nOptimization Result:")
    print(f"Success: {res.success}")
    print(f"Message: {res.message}")
    print(f"Optimal Angles (deg): {np.degrees(res.x)}")
    print(f"Minimum Total Cost (dimensionless): {res.fun}")

    V_star = 1024.0
    print(f"Minimum Total Cost (m/s): {res.fun * V_star:.2f}")

    # Compare with baseline
    baseline_cost = cost_function([0, 0, 0])
    print(f"Baseline Cost (m/s): {baseline_cost * V_star:.2f}")
    print(f"Improvement: {(baseline_cost - res.fun)/baseline_cost * 100:.2f}%")


if __name__ == "__main__":
    optimize_formation()
