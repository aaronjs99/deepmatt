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

def find_transfer():
    print("Generating stable manifold points...")
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
    
    # Generate points on the stable manifold
    # We use more points to increase chance of finding a good transfer
    manifold_points = generate_stable_manifold(initial_point_opt, num_points=50)
    
    print(f"Generated {len(manifold_points)} manifold points. integrating backwards...")
    
    transfers = []
    earth_pos = np.array([-mu, 0, 0]) # Earth position in rotating frame
    
    # Integration parameters
    t_span = (0, -15.0) # Integrate backwards for ~15 dimensionless time units (approx 65 days)
    t_eval = np.linspace(0, -15.0, 3000)
    
    best_dist = np.inf
    best_transfer = None
    
    for i, point in enumerate(manifold_points):
        sol = solve_ivp(
            halo_dynamics,
            t_span,
            point,
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12
        )
        
        # Check distance to Earth at each step
        trajectory = sol.y.T
        dists = np.linalg.norm(trajectory[:, :3] - earth_pos, axis=1)
        min_dist = np.min(dists)
        
        if min_dist < best_dist:
            best_dist = min_dist
            best_transfer = (sol.t, trajectory)
            
        if i % 10 == 0:
            print(f"Processed {i}/{len(manifold_points)} points. Best dist so far: {best_dist:.4f}")

    print(f"Best transfer found! Min distance to Earth: {best_dist:.4f} dimensionless units")
    
    # Plotting
    if best_transfer:
        t, state = best_transfer
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot transfer trajectory
        ax.plot(state[:, 0].real, state[:, 1].real, state[:, 2].real, 'b-', label='Transfer Trajectory')
        
        # Plot Earth
        ax.scatter([-mu], [0], [0], color='blue', s=100, label='Earth')
        
        # Plot Moon (approximate)
        ax.scatter([1-mu], [0], [0], color='gray', s=50, label='Moon')
        
        # Plot Halo Orbit injection point
        ax.scatter([state[0, 0].real], [state[0, 1].real], [state[0, 2].real], color='green', marker='*', s=100, label='Halo Injection')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Manifold Transfer to L1 Halo Orbit\nMin Dist to Earth: {best_dist:.4f}')
        ax.legend()
        
        plt.savefig("results/transfer_3d.png")
        plt.close()
        print("Saved transfer plot to results/transfer_3d.png")

def main():
    find_transfer()

if __name__ == "__main__":
    main()
