import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from cislune.stability import get_stable_eigenvector, halo_dynamics, mu

def poincare_map():
    print("Generating Poincaré Map...")
    
    # L1 Halo orbit initial point
    initial_point = np.array(
        [
            8.2352859616538130e-01,
            6.3030050026113780e-03,
            1.1041843760609028e-06,
            5.4958758023583450e-03,
            1.2551080993197510e-01,
            -2.4522163866711757e-07,
        ]
    )
    
    eigvec = get_stable_eigenvector(initial_point)
    if eigvec is None:
        print("Could not find stable eigenvector.")
        return

    # Event function to detect crossing z=0 plane
    def z_crossing(t, y):
        return y[2]
    z_crossing.terminal = False
    z_crossing.direction = 0 # Detect both directions

    # Generate trajectories
    perturbations = np.linspace(-1e-4, 1e-4, 50)
    intersections = []

    for dr in perturbations:
        point = initial_point + dr * eigvec / np.linalg.norm(eigvec)
        
        sol = solve_ivp(
            halo_dynamics,
            (0, -20.0), # Integrate backwards
            point,
            method="DOP853",
            events=z_crossing,
            rtol=1e-9,
            atol=1e-12
        )
        
        if len(sol.t_events[0]) > 0:
            # Collect (x, vx) at crossing
            for i in range(len(sol.t_events[0])):
                # y_events is a list of arrays, one for each event type
                state = sol.y_events[0][i]
                intersections.append([state[0], state[3]]) # x, vx

    intersections = np.array(intersections)
    
    if len(intersections) > 0:
        plt.figure(figsize=(10, 8))
        plt.scatter(intersections[:, 0], intersections[:, 1], s=1, c='blue')
        plt.xlabel('x (dimensionless)')
        plt.ylabel('vx (dimensionless)')
        plt.title('Poincaré Map (z=0 Section) of Stable Manifold')
        plt.grid(True)
        plt.savefig("results/poincare_map.png")
        plt.close()
        print("Saved Poincaré map to results/poincare_map.png")
    else:
        print("No intersections found.")

def main():
    poincare_map()

if __name__ == "__main__":
    main()
