# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar, differential_evolution

from cislune.params import mu, L1_HALO_INITIAL_STATE, EARTH_POS, MOON_POS
from cislune.gravdyn import cr3bp_dynamics as halo_dynamics, get_stable_eigenvector
from cislune.plotting import plot_trajectory_3d

def find_transfer():
    print("Optimizing transfer trajectory...")
    # L1 Halo orbit initial point (approximate)
    initial_point_opt = np.array(L1_HALO_INITIAL_STATE)
    
    eigvec = get_stable_eigenvector(initial_point_opt)
    if eigvec is None:
        print("Could not find stable eigenvector.")
        return

    t_span = (0, -15.0)
    t_eval = np.linspace(0, -15.0, 3000)

    def objective(dr):
        point = initial_point_opt + dr * eigvec / np.linalg.norm(eigvec)
        sol = solve_ivp(
            halo_dynamics,
            t_span,
            point,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12
        )
        trajectory = sol.y.T
        dists = np.linalg.norm(trajectory[:, :3] - EARTH_POS, axis=1)
        return np.min(dists)

    # Search range for perturbation
    point_range = 0.000000010000
    
    # We optimize over dr
    res = minimize_scalar(objective, bounds=(-point_range, point_range), method='bounded')
    
    if res.success:
        best_dr = res.x
        min_dist = res.fun
        print(f"Optimization successful! Best dr: {best_dr}")
        print(f"Minimum distance to Earth: {min_dist:.6f} dimensionless units")
        
        # Re-run best trajectory for plotting
        point = initial_point_opt + best_dr * eigvec / np.linalg.norm(eigvec)
        sol = solve_ivp(
            halo_dynamics,
            t_span,
            point,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12
        )
        state = sol.y.T
        
        plot_trajectory_3d(
            state, 
            f'Optimized Manifold Transfer to L1 Halo Orbit\nMin Dist to Earth: {min_dist:.6f}',
            "transfer_earth.png",
            other_points=[{'pos': state[0, :3], 'color': 'green', 'marker': '*', 's': 100, 'label': 'Halo Injection'}]
        )
    else:
        print("Optimization failed.")


def optimize_leo_transfer():
    print("Optimizing transfer to LEO (Radius ~6700 km)...")
    
    # L1 Halo orbit initial point
    initial_point_opt = np.array(L1_HALO_INITIAL_STATE)
    
    target_radius = 6700 / 384400.0 # Dimensionless LEO radius
    print(f"Target Radius: {target_radius:.6f}")

    def objective(params):
        tau, dr = params
        
        # 1. Propagate along Halo orbit for time tau
        sol_halo = solve_ivp(
            halo_dynamics,
            (0, tau),
            initial_point_opt,
            method="DOP853",
            rtol=1e-9,
            atol=1e-12
        )
        point_on_halo = sol_halo.y[:, -1]
        
        # 2. Get stable eigenvector at this point
        eigvec = get_stable_eigenvector(point_on_halo)
        if eigvec is None:
            return 10.0 # Penalty if unstable
            
        # 3. Perturb onto manifold
        point_manifold = point_on_halo + dr * eigvec / np.linalg.norm(eigvec)
        
        # 4. Integrate backwards to Earth
        sol_transfer = solve_ivp(
            halo_dynamics,
            (0, -15.0),
            point_manifold,
            method="DOP853",
            t_eval=np.linspace(0, -15.0, 1000),
            rtol=1e-9,
            atol=1e-12
        )
        
        # 5. Calculate perigee deviation
        trajectory = sol_transfer.y.T
        dists = np.linalg.norm(trajectory[:, :3] - EARTH_POS, axis=1)
        min_dist = np.min(dists)
        
        return abs(min_dist - target_radius)

    # Bounds: tau [0, 3.0], dr [-1e-2, 1e-2]
    bounds = [(0.0, 3.0), (-1e-2, 1e-2)]
    
    # Use differential_evolution for global optimization
    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=20, popsize=10, tol=1e-3)
    
    print(res)
    if not res.success:
        print("Optimization did not converge to tolerance, but using best found solution.")

    print("Optimization finished.")
    best_tau, best_dr = res.x
    print(f"Best tau: {best_tau:.4f}")
    print(f"Best dr: {best_dr}")
    print(f"Final Objective (Radius Error): {res.fun:.6f}")
    
    # Re-run for plotting
    sol_halo = solve_ivp(halo_dynamics, (0, best_tau), initial_point_opt, method="DOP853", rtol=1e-9, atol=1e-12)
    point_on_halo = sol_halo.y[:, -1]
    eigvec = get_stable_eigenvector(point_on_halo)
    point_manifold = point_on_halo + best_dr * eigvec / np.linalg.norm(eigvec)
    sol_transfer = solve_ivp(halo_dynamics, (0, -15.0), point_manifold, method="DOP853", t_eval=np.linspace(0, -15.0, 3000), rtol=1e-9, atol=1e-12)
    state = sol_transfer.y.T
    dists = np.linalg.norm(state[:, :3] - EARTH_POS, axis=1)
    final_min_dist = np.min(dists)
    print(f"Final Perigee Radius: {final_min_dist:.6f} (Target: {target_radius:.6f})")

    # Delta-V Calculation
    r_perigee = state[-1, :3]
    v_perigee_rot = state[-1, 3:]
    
    # v_inertial = v_rot + cross([0,0,1], r)
    v_inertial = v_perigee_rot + np.cross([0, 0, 1], r_perigee)
    v_perigee_mag = np.linalg.norm(v_inertial)
    
    # Circular velocity at this radius around Earth
    r_mag = np.linalg.norm(r_perigee - EARTH_POS)
    v_circ = np.sqrt((1 - mu) / r_mag)
    
    delta_v_leo = abs(v_perigee_mag - v_circ)
    
    print(f"--- Delta-V Analysis ---")
    print(f"Perigee Radius: {r_mag * 384400:.2f} km")
    print(f"Velocity at Perigee (Inertial): {v_perigee_mag * 1024.6:.2f} m/s") # 1024.6 m/s is approx velocity unit
    print(f"Circular Velocity at Perigee: {v_circ * 1024.6:.2f} m/s")
    print(f"LEO Injection Delta-V: {delta_v_leo * 1024.6:.2f} m/s")

    plot_trajectory_3d(
        state,
        f'Optimized Transfer to LEO\nPerigee: {final_min_dist:.6f}\nDelta-V: {delta_v_leo * 1024.6:.2f} m/s',
        "transfer_leo.png"
    )

def optimize_time_transfer():
    print("Optimizing Time-Optimal Transfer to Earth...")
    
    # L1 Halo orbit initial point
    initial_point_opt = np.array(L1_HALO_INITIAL_STATE)
    
    target_dist = 0.05 # Define "Earth vicinity" as within this distance

    def objective(params):
        tau, dr, tof = params
        
        # 1. Propagate along Halo orbit for time tau
        sol_halo = solve_ivp(
            halo_dynamics,
            (0, tau),
            initial_point_opt,
            method="DOP853",
            rtol=1e-9,
            atol=1e-12
        )
        point_on_halo = sol_halo.y[:, -1]
        
        # 2. Get stable eigenvector
        eigvec = get_stable_eigenvector(point_on_halo)
        if eigvec is None:
            return 100.0 # Penalty
            
        # 3. Perturb onto manifold
        point_manifold = point_on_halo + dr * eigvec / np.linalg.norm(eigvec)
        
        # 4. Integrate backwards for TOF
        sol_transfer = solve_ivp(
            halo_dynamics,
            (0, -tof),
            point_manifold,
            method="DOP853",
            rtol=1e-9,
            atol=1e-12
        )
        
        final_state = sol_transfer.y[:, -1]
        dist_to_earth = np.linalg.norm(final_state[:3] - EARTH_POS)
        
        # Objective: Minimize TOF, but must reach Earth vicinity
        # Penalty method: J = TOF + Penalty * max(0, dist - target_dist)
        penalty = 1000.0 * max(0, dist_to_earth - target_dist)
        
        return tof + penalty

    # Bounds: tau [0, 3.0], dr [-1e-2, 1e-2], tof [0, 20.0]
    bounds = [(0.0, 3.0), (-1e-2, 1e-2), (5.0, 20.0)]
    
    res = differential_evolution(objective, bounds, strategy='best1bin', maxiter=20, popsize=10, tol=1e-3)
    
    print(res)
    if res.success or True: # Proceed with best result
        best_tau, best_dr, best_tof = res.x
        print(f"Best tau: {best_tau:.4f}")
        print(f"Best dr: {best_dr}")
        print(f"Best TOF: {best_tof:.4f}")
        
        # Re-run for plotting
        sol_halo = solve_ivp(halo_dynamics, (0, best_tau), initial_point_opt, method="DOP853", rtol=1e-9, atol=1e-12)
        point_on_halo = sol_halo.y[:, -1]
        eigvec = get_stable_eigenvector(point_on_halo)
        point_manifold = point_on_halo + best_dr * eigvec / np.linalg.norm(eigvec)
        sol_transfer = solve_ivp(halo_dynamics, (0, -best_tof), point_manifold, method="DOP853", t_eval=np.linspace(0, -best_tof, 1000), rtol=1e-9, atol=1e-12)
        state = sol_transfer.y.T
        
        plot_trajectory_3d(
            state,
            f'Time-Optimal Transfer\nTOF: {best_tof:.4f} (dimensionless)',
            "transfer_fast.png",
            other_points=[{'pos': state[0, :3], 'color': 'green', 'marker': '*', 's': 100, 'label': 'Halo Injection'}]
        )

def main():
    # find_transfer()
    # optimize_leo_transfer()
    optimize_time_transfer()

if __name__ == "__main__":
    main()
