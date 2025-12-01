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
from cislune.stability import generate_stable_manifold, halo_dynamics, mu

from scipy.optimize import minimize_scalar
from cislune.stability import get_stable_eigenvector, halo_dynamics, mu

def find_transfer():
    print("Optimizing transfer trajectory...")
    # L1 Halo orbit initial point (approximate)
    initial_point_opt = np.array(
        [
            8.2352859616538130e-01,
            6.3030050026113780e-03,
            1.1041843760609028e-06,
            5.4958758023583450e-03,
            1.2551080993197510e-01,
            -2.4522163866711757e-07,
        ]
    )
    
    eigvec = get_stable_eigenvector(initial_point_opt)
    if eigvec is None:
        print("Could not find stable eigenvector.")
        return

    earth_pos = np.array([-mu, 0, 0])
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
        dists = np.linalg.norm(trajectory[:, :3] - earth_pos, axis=1)
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
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot transfer trajectory
        ax.plot(state[:, 0].real, state[:, 1].real, state[:, 2].real, 'b-', label='Optimized Transfer')
        
        # Plot Earth
        ax.scatter([-mu], [0], [0], color='blue', s=100, label='Earth')
        
        # Plot Moon (approximate)
        ax.scatter([1-mu], [0], [0], color='gray', s=50, label='Moon')
        
        # Plot Halo Orbit injection point
        ax.scatter([state[0, 0].real], [state[0, 1].real], [state[0, 2].real], color='green', marker='*', s=100, label='Halo Injection')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Optimized Manifold Transfer to L1 Halo Orbit\nMin Dist to Earth: {min_dist:.6f}')
        ax.legend()
        
        plt.savefig("results/transfer_optimized_3d.png")
        plt.close()
        print("Saved optimized transfer plot to results/transfer_optimized_3d.png")
    else:
        print("Optimization failed.")

from scipy.optimize import minimize, differential_evolution

def optimize_leo_transfer():
    print("Optimizing transfer to LEO (Radius ~6700 km)...")
    
    # L1 Halo orbit initial point
    initial_point_opt = np.array(
        [
            8.2352859616538130e-01,
            6.3030050026113780e-03,
            1.1041843760609028e-06,
            5.4958758023583450e-03,
            1.2551080993197510e-01,
            -2.4522163866711757e-07,
        ]
    )
    
    earth_pos = np.array([-mu, 0, 0])
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
        dists = np.linalg.norm(trajectory[:, :3] - earth_pos, axis=1)
        min_dist = np.min(dists)
        
        return abs(min_dist - target_radius)

    # Initial guess: tau=0, dr=best_dr from previous step
    initial_guess = [0.0, -2.36e-9] 
    
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
    # ... (Re-run logic similar to objective function)
    sol_halo = solve_ivp(halo_dynamics, (0, best_tau), initial_point_opt, method="DOP853", rtol=1e-9, atol=1e-12)
    point_on_halo = sol_halo.y[:, -1]
    eigvec = get_stable_eigenvector(point_on_halo)
    point_manifold = point_on_halo + best_dr * eigvec / np.linalg.norm(eigvec)
    sol_transfer = solve_ivp(halo_dynamics, (0, -15.0), point_manifold, method="DOP853", t_eval=np.linspace(0, -15.0, 3000), rtol=1e-9, atol=1e-12)
    state = sol_transfer.y.T
    dists = np.linalg.norm(state[:, :3] - earth_pos, axis=1)
    final_min_dist = np.min(dists)
    print(f"Final Perigee Radius: {final_min_dist:.6f} (Target: {target_radius:.6f})")

    # Delta-V Calculation
    # 1. Insertion Delta-V at Halo Orbit (assuming we arrive with 0 relative velocity for simplicity, or just matching state)
    # In reality, we are on the manifold which asymptotically approaches the Halo orbit, so theoretically Delta-V is 0.
    # However, practically we might need a small burn to stabilize.
    # Here we calculate the departure Delta-V from LEO.
    
    # Velocity at perigee (from transfer trajectory)
    # The state is [x, y, z, vx, vy, vz]
    # We need to convert to inertial frame or use rotating frame velocity properly.
    # In rotating frame, v_inertial = v_rotating + omega x r
    # omega = [0, 0, 1]
    
    r_perigee = state[-1, :3]
    v_perigee_rot = state[-1, 3:]
    
    # v_inertial = v_rot + cross([0,0,1], r)
    v_inertial = v_perigee_rot + np.cross([0, 0, 1], r_perigee)
    v_perigee_mag = np.linalg.norm(v_inertial)
    
    # Circular velocity at this radius around Earth
    # v_circ = sqrt(mu_earth / r)
    # In dimensionless units, G(M1+M2) = 1.
    # mu_earth = 1 - mu
    r_mag = np.linalg.norm(r_perigee - earth_pos)
    v_circ = np.sqrt((1 - mu) / r_mag)
    
    delta_v_leo = abs(v_perigee_mag - v_circ)
    
    print(f"--- Delta-V Analysis ---")
    print(f"Perigee Radius: {r_mag * 384400:.2f} km")
    print(f"Velocity at Perigee (Inertial): {v_perigee_mag * 1024.6:.2f} m/s") # 1024.6 m/s is approx velocity unit
    print(f"Circular Velocity at Perigee: {v_circ * 1024.6:.2f} m/s")
    print(f"LEO Injection Delta-V: {delta_v_leo * 1024.6:.2f} m/s")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state[:, 0].real, state[:, 1].real, state[:, 2].real, 'b-', label='LEO Transfer')
    ax.scatter([-mu], [0], [0], color='blue', s=100, label='Earth')
    ax.scatter([1-mu], [0], [0], color='gray', s=50, label='Moon')
    ax.set_title(f'Optimized Transfer to LEO\nPerigee: {final_min_dist:.6f}\nDelta-V: {delta_v_leo * 1024.6:.2f} m/s')
    ax.legend()
    plt.savefig("results/transfer_leo_3d.png")
    plt.close()
    print("Saved plot to results/transfer_leo_3d.png")

def main():
    # find_transfer()
    optimize_leo_transfer()

if __name__ == "__main__":
    main()
