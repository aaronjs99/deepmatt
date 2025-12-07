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

import csv
import numpy as np
from gekko import GEKKO
import scipy.optimize as opt
import scipy.integrate as intg
import matplotlib.pyplot as plt

from deepmatt.core.params import mu, L1_HALO_INITIAL_STATE
from deepmatt.dynamics.cr3bp import (
    cr3bp_dynamics as halo_dynamics,
    linearized_dynamics,
    get_stable_eigenvector,
)
from deepmatt.core.plotting import plot_trajectory_3d

save_orbit = False


def dist_to_equilibrium(point, equilibrium_points):
    min_dist = np.inf
    for eq in equilibrium_points:
        dist = np.linalg.norm(point - eq)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def stability_check(
    platform, dynamics, initial_point, equilibrium_points, threshold, time_params
):
    f = dynamics
    if platform == "gekko":
        m = GEKKO(remote=True, server="https://apmonitor.com")
        m.time = np.linspace(
            time_params["initial_time"],
            time_params["final_time"],
            time_params["num_steps"],
        )

        x = [m.Var(value=initial_point[i]) for i in range(len(initial_point))]
        t = m.Var(value=time_params["initial_time"])

        equations = []
        equations.append(t.dt() == time_params["dt"])
        for i in range(len(x)):
            equations.append(x[i].dt() == f(t, x, m=m)[i])
        m.Equations(equations)

        m.options.IMODE = 4
        m.options.MAX_ITER = 1000
        m.options.SOLVER = 2
        m.Minimize(0.0)
        m.solve(disp=True)

        x = np.transpose(np.array([x[i].value for i in range(len(x))]))
    elif platform == "scipy":
        x = np.array(initial_point)
        t = np.linspace(
            time_params["initial_time"],
            time_params["final_time"],
            time_params["num_steps"],
        )
        x = intg.solve_ivp(
            fun=f,
            t_span=(time_params["initial_time"], time_params["final_time"]),
            y0=x,
            method="DOP853",
            t_eval=t,
        )
        t = x.t
        x = x.y
        x = np.transpose(x)

    final_state = np.array([x[-1][i] for i in range(len(x[0]))])
    print(f"Final State: {final_state}")

    return x


def generate_stable_manifold(
    initial_point_opt, num_points=11, rep_num_points=100, max_reps=10000
):
    stable_init_points = []
    stable_manifold = []
    point_range = 0.000000010000
    eig_tol = 1.0e-5

    # Initial generation around the equilibrium point
    eigvec = get_stable_eigenvector(initial_point_opt, eig_tol)

    if eigvec is not None:
        stable_init_points.append(initial_point_opt)
        for dr in np.linspace(-point_range, point_range, num_points):
            if dr == 0:
                continue
            stable_manifold.append(
                initial_point_opt + dr * eigvec / np.linalg.norm(eigvec)
            )

    return stable_manifold


def main():
    equilibrium_points = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    threshold = 1e-3
    dt = 0.001
    initial_time = 0.0
    final_time = 2.75 * 1.5  # 2.7427

    all_points = []

    orbit = "L1"
    halo_points = []  # Initialize halo_points here
    if not save_orbit:
        try:
            with open("data/halo_orbit_points.csv", "r") as file:
                reader = csv.reader(file)
                for row in reader:
                    halo_points.append([float(row[i]) for i in range(len(row))])
        except FileNotFoundError:
            print("halo_orbit_points.csv not found. Skipping orbit loading.")
            halo_points = []

    # Use centralized initial state
    initial_point_opt = np.array(L1_HALO_INITIAL_STATE)

    mode = "stable"
    # dr_comp logic removed as it's replaced by eigenvector method in other modules
    # But kept here for legacy support if needed, or we can just use the eigenvector method.
    # For now, let's keep the loop structure but use the eigenvector.

    eigvec = get_stable_eigenvector(initial_point_opt)
    if eigvec is None:
        print("No stable eigenvector found.")
        return

    dr_comp = eigvec / np.linalg.norm(eigvec)

    midpoint = 0.0
    point_range = 0.000000010000
    num_points = 11

    if True:
        for dr in np.linspace(
            midpoint - point_range / 2, midpoint + point_range / 2, num_points
        ):
            initial_point = initial_point_opt + dr * dr_comp
            all_points.append(
                stability_check(
                    platform="scipy",
                    dynamics=halo_dynamics,
                    initial_point=initial_point,
                    equilibrium_points=equilibrium_points,
                    threshold=threshold,
                    time_params={
                        "initial_time": initial_time,
                        "final_time": final_time,
                        "dt": dt,
                        "num_steps": int((final_time - initial_time) / dt),
                    },
                )
            )

    if save_orbit:
        with open("data/halo_orbit_points.csv", "w", newline="") as file:
            writer = csv.writer(file)
            for point in all_points[0]:
                writer.writerow(point)

    # Plotting using shared module
    # We need to flatten the list of trajectories into a single array or handle multiple trajectories
    # plotting.py handles one trajectory array. We can loop or modify plotting.py.
    # For now, let's just plot the first one or modify plotting.py to accept list of trajectories?
    # Actually, plotting.py takes 'states'.

    # Let's just use the original plotting logic for now but clean it up,
    # OR use plotting.py for each trajectory? No that would create many files.
    # Let's stick to local plotting for this specific multi-trajectory visualization
    # OR update plotting.py to handle multiple trajectories.
    # Given the instruction "try it make it as modular as possible", I should probably use plotting.py.
    # But plotting.py is designed for single trajectory + points.

    # Let's keep the local plotting for this specific visualization as it's quite custom (multiple orbits).

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for orbit in all_points:
        ax.plot(
            [orbit[k][0] for k in range(len(orbit))],
            [orbit[k][1] for k in range(len(orbit))],
            [orbit[k][2] for k in range(len(orbit))],
            "b-",
        )
    if halo_points:
        ax.plot(
            [halo_points[k][0] for k in range(len(halo_points))],
            [halo_points[k][1] for k in range(len(halo_points))],
            [halo_points[k][2] for k in range(len(halo_points))],
            "k-",
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_zlabel("Z")
    plt.savefig("results/analysis/stability_3d.png")
    plt.close()
    plt.close()


if __name__ == "__main__":
    main()
