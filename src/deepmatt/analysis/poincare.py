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

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import (
    cr3bp_dynamics as halo_dynamics,
    get_stable_eigenvector,
)


def poincare_map():
    print("Generating Poincaré Map...")

    # L1 Halo orbit initial point
    initial_point = np.array(L1_HALO_INITIAL_STATE)

    eigvec = get_stable_eigenvector(initial_point)
    if eigvec is None:
        print("Could not find stable eigenvector.")
        return

    # Event function to detect crossing z=0 plane
    def z_crossing(t, y):
        return y[2]

    z_crossing.terminal = False
    z_crossing.direction = 0  # Detect both directions

    # Generate trajectories
    # Increase density for better resolution
    perturbations = np.linspace(-1e-4, 1e-4, 500)
    intersections = []

    for dr in perturbations:
        point = initial_point + dr * eigvec / np.linalg.norm(eigvec)

        sol = solve_ivp(
            halo_dynamics,
            (0, -20.0),  # Integrate backwards
            point,
            method="DOP853",
            events=z_crossing,
            rtol=1e-9,
            atol=1e-12,
        )

        if len(sol.t_events[0]) > 0:
            # Collect (x, vx) at crossing
            for i, state in enumerate(sol.y_events[0]):
                # state is the full state vector at the event
                intersections.append([state[0], state[3], i])  # x, vx, crossing_number

    intersections = np.array(intersections)

    if len(intersections) > 0:
        plt.figure(figsize=(12, 10))
        # Color by crossing number to show structure
        sc = plt.scatter(
            intersections[:, 0],
            intersections[:, 1],
            c=intersections[:, 2],
            cmap="viridis",
            s=2,
            alpha=0.8,
        )
        plt.colorbar(sc, label="Crossing Number")
        plt.xlabel("x (dimensionless)")
        plt.ylabel("vx (dimensionless)")
        plt.title("Poincaré Map (z=0 Section) of Stable Manifold")
        plt.grid(True, alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.savefig("results/poincare/poincare_map.png", dpi=300)
        plt.close()
        print("Saved Poincaré map to results/poincare/poincare_map.png")
    else:
        print("No intersections found.")


def main():
    poincare_map()


if __name__ == "__main__":
    main()
