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
from scipy.integrate import solve_ivp

"""
Mission Design and Analysis Module.

Calculates total delta-V budgets for the formation flying mission, including:
- LEO to Halo transfer injection
- Halo Orbit Insertion (HOI)
- Formation deployment maneuvers
- Station-keeping estimation
"""

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import (
    cr3bp_dynamics as halo_dynamics,
    get_stable_eigenvector,
)
from deepmatt.analysis.transfer import optimize_leo_transfer


def calculate_mission_cost():
    print("Calculating Mission Design & Deployment Costs...")

    # 1. Get Transfer Trajectory (LEO -> Halo)
    # This runs the optimization (might take a moment)
    print("Finding optimal transfer from LEO...")
    tau, dr, dv_leo = optimize_leo_transfer()

    # 2. Propagate to Insertion Point (HOI)
    initial_point = np.array(L1_HALO_INITIAL_STATE)

    # Propagate along Halo to injection point
    sol_halo = solve_ivp(
        halo_dynamics, (0, tau), initial_point, method="DOP853", rtol=1e-9, atol=1e-12
    )
    x_halo_injection = sol_halo.y[:, -1]  # State of Chief at injection

    # Get Manifold State (Transfer Vehicle Arrival)
    eigvec = get_stable_eigenvector(x_halo_injection)
    x_transfer_arrival = x_halo_injection + dr * eigvec / np.linalg.norm(eigvec)

    # Calculate HOI Delta-V for Chief
    # V_transfer_arrival is the velocity of the manifold at the injection point?
    # No, x_transfer_arrival IS the point on the manifold.
    # The velocity match is: V_halo - V_transfer
    # But V_transfer is asymptotic to V_halo.
    # The "Manifold Insertion" usually implies zero Delta-V if you wait infinite time.
    # But we arrive at a specific point.
    # Actually, the "dr" perturbation puts us on the stable manifold.
    # If we are ON the stable manifold, we will naturally drift into the Halo orbit.
    # So theoretically, HOI Delta-V is ZERO!
    # However, in practice, we might need a small correction to hop off the manifold exactly onto the periodic orbit.
    # Let's assume Delta-V HOI ~ 0 (or very small station keeping).
    # BUT, we calculated `dv_leo` at Earth.

    print("\n--- Mission Delta-V Summary ---")
    V_star = 1024.0
    print(f"LEO Injection Delta-V: {dv_leo * V_star:.2f} m/s")
    print(f"Halo Orbit Insertion (HOI) Delta-V: ~0 m/s (Asymptotic Capture)")

    # 3. Deployment Cost for Formation
    # Assume Mother Ship arrives at x_halo_injection.
    # Deputies need to move to x_halo_injection + offset.
    # Let's estimate the cost to move 10 km in 1 day (approx 0.23 dimensionless time).

    dist_km = 10.0
    dist_dim = dist_km / 384400.0
    transfer_time_days = 1.0
    T = transfer_time_days * (2 * np.pi / 27.3)  # dimensionless time

    # Simple approximation: Delta-V = Distance / Time
    # (Neglecting gravity gradient for short transfer)
    dv_deploy_dim = dist_dim / T
    dv_deploy_ms = dv_deploy_dim * V_star

    # Two burns (start and stop)? -> 2 * dv
    total_deploy_cost = 2 * dv_deploy_ms

    print(f"\n--- Formation Deployment Cost (Estimate) ---")
    print(f"Target Separation: {dist_km} km")
    print(f"Transfer Time: {transfer_time_days} days")
    print(f"Deployment Delta-V (per agent): ~{total_deploy_cost:.2f} m/s")

    print("\n--- Total Mission Cost per Agent ---")
    print(f"Chief: {dv_leo * V_star:.2f} m/s")
    print(f"Deputies/Officers: {dv_leo * V_star + total_deploy_cost:.2f} m/s")


def main():
    calculate_mission_cost()


if __name__ == "__main__":
    main()
