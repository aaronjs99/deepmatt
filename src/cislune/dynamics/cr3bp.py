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
from gekko import GEKKO
from typing import Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from cislune.core.utils import load_config, Vector, integrate_simulator

import os
# Try to find config file
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
config_path = os.path.join(base_dir, "config", "default.yaml")
if not os.path.exists(config_path):
    # Fallback for when installed as package (if config is packaged)
    # For now, just try local config.yaml
    config_path = "config.yaml"

config = load_config(config_path)

mu_EM = config["mu_M"] / (config["mu_E"] + config["mu_M"])
p_E = Vector(-mu_EM * config["dist_EM"], 0.0, 0.0)
p_M = Vector((1 - mu_EM) * config["dist_EM"], 0.0, 0.0)


def acc(t, p: Union[tuple, Vector], *args, **kwargs):
    mu_E = config["mu_E"]
    mu_M = config["mu_M"]
    p_E = config["derived"]["p_E"]
    p_M = config["derived"]["p_M"]
    if isinstance(p, tuple):
        m = kwargs["m"]
        a_E = mu_E / (
            m.sqrt((p[0] - p_E.x) ** 2 + (p[1] - p_E.y) ** 2 + (p[2] - p_E.z) ** 2) ** 3
        )
        a_M = mu_M / (
            m.sqrt((p[0] - p_M.x) ** 2 + (p[1] - p_M.y) ** 2 + (p[2] - p_M.z) ** 2) ** 3
        )
        a_X = -(a_E * (p[0] - p_E.x) + a_M * (p[0] - p_M.x))
        a_Y = -(a_E * (p[1] - p_E.y) + a_M * (p[1] - p_M.y))
        a_Z = -(a_E * (p[2] - p_E.z) + a_M * (p[2] - p_M.z))
        return (a_X, a_Y, a_Z)
    elif isinstance(p, Vector):
        a_E = mu_E / (
            np.linalg.norm(np.array([p.x - p_E.x, p.y - p_E.y, p.z - p_E.z])) ** 3
        )
        a_M = mu_M / (
            np.linalg.norm(np.array([p.x - p_M.x, p.y - p_M.y, p.z - p_M.z])) ** 3
        )
        a_X = -(a_E * (p.x - p_E.x) + a_M * (p.x - p_M.x))
        a_Y = -(a_E * (p.y - p_E.y) + a_M * (p.y - p_M.y))
        a_Z = -(a_E * (p.z - p_E.z) + a_M * (p.z - p_M.z))
        return Vector(a_X, a_Y, a_Z)


def gekko_simulator(
    p_0: Vector, v_0: Vector = Vector(0.0, 0.0, 0.0), dt=1, num_steps=1000
):
    m = GEKKO(remote=True)
    m.time = np.linspace(0, dt * num_steps, num_steps)

    px = m.Var(value=p_0.x, fixed_initial=True)
    py = m.Var(value=p_0.y, fixed_initial=True)
    pz = m.Var(value=p_0.z, fixed_initial=True)

    vx = m.Var(value=v_0.x, fixed_initial=True)
    vy = m.Var(value=v_0.y, fixed_initial=True)
    vz = m.Var(value=v_0.z, fixed_initial=True)

    ax = m.Var()
    ay = m.Var()
    az = m.Var()

    m.Equation(px.dt() == vx)
    m.Equation(py.dt() == vy)
    m.Equation(pz.dt() == vz)

    m.Equation(vx.dt() == ax)
    m.Equation(vy.dt() == ay)
    m.Equation(vz.dt() == az)

    m.Equation(ax == acc((px, py, pz), m=m)[0])
    m.Equation(ay == acc((px, py, pz), m=m)[1])
    m.Equation(az == acc((px, py, pz), m=m)[2])

    m.options.IMODE = 4
    m.options.SOLVER = 1
    m.options.MAX_ITER = 1000
    m.solve(disp=True)

    return [Vector(px[i], py[i], pz[i]) for i in range(num_steps)]


if __name__ == "__main__":
    p_0 = Vector(2.00e8 - mu_EM * config["dist_EM"], 6.678e6, 0.00e8)
    v_0 = Vector(0.00e3, 1.000e3, 1.00e3)
    dt = 1.0e1
    num_steps = int(1.0e6)

    positions = integrate_simulator(
        sys=acc, p_0=p_0, v_0=v_0, t_0=0.0, dt=dt, num_steps=num_steps
    )
    # positions = gekko_simulator(p_0, v_0, dt, num_steps)

    x_coords = [position.x for position in positions]
    y_coords = [position.y for position in positions]
    z_coords = [position.z for position in positions]
    dist_E = [
        np.linalg.norm(np.array([x - p_E.x, y - p_E.y, z - p_E.z]))
        for x, y, z in zip(x_coords, y_coords, z_coords)
    ]
    dist_M = [
        np.linalg.norm(np.array([x - p_M.x, y - p_M.y, z - p_M.z]))
        for x, y, z in zip(x_coords, y_coords, z_coords)
    ]
    print(
        np.min(dist_E),
        np.argmin(dist_E),
        (
            positions[np.argmin(dist_E)].x,
            positions[np.argmin(dist_E)].y,
            positions[np.argmin(dist_E)].z,
        ),
    )
    print(
        np.min(dist_M),
        np.argmin(dist_M),
        (
            positions[np.argmin(dist_M)].x,
            positions[np.argmin(dist_M)].y,
            positions[np.argmin(dist_M)].z,
        ),
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_coords, y_coords, z_coords)

    ax.scatter(p_E.x, p_E.y, p_E.z, c="blue", marker="o", label="Earth")
    ax.scatter(p_M.x, p_M.y, p_M.z, c="grey", marker="o", label="Moon")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Cislunar Orbit Simulation")

    plt.show()
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(np.arange(0, dt * num_steps, dt), x_coords)
    axes[0].set_title("X")

    axes[1].plot(np.arange(0, dt * num_steps, dt), y_coords)
    axes[1].set_title("Y")

    axes[2].plot(np.arange(0, dt * num_steps, dt), z_coords)
    axes[2].set_xlabel("t")
    axes[2].set_title("Z")

    plt.show()

# --- Refactored Dynamics ---
from cislune.core.params import mu

def cr3bp_dynamics(t, x, *args, **kwargs):
    """
    Circular Restricted Three-Body Problem (CR3BP) dynamics.
    """
    if "m" in kwargs:
        m = kwargs["m"]
    else:
        m = np
    r1 = m.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
    r2 = m.sqrt((x[0] - 1 + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
    return [
        x[3],
        x[4],
        x[5],
        x[0] + 2 * x[4] - (1 - mu) * (x[0] + mu) / r1**3 - mu * (x[0] - 1 + mu) / r2**3,
        x[1] - 2 * x[3] - (1 - mu) * x[1] / r1**3 - mu * x[1] / r2**3,
        -(1 - mu) * x[2] / r1**3 - mu * x[2] / r2**3,
    ]

def linearized_dynamics(x):
    """
    Linearized dynamics matrix (Jacobian) for CR3BP.
    """
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3, 4] = 2
    A[4, 3] = -2
    
    r1 = np.sqrt((x[0] + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
    r2 = np.sqrt((x[0] - 1 + mu) ** 2 + x[1] ** 2 + x[2] ** 2)
    
    # Partial derivatives of potential U w.r.t position
    # Uxx, Uyy, Uzz, Uxy, Uxz, Uyz
    # Simplified for collinear points (y=z=0) but full form needed generally
    # Using the form from stability.py
    
    # Full Jacobian implementation
    # Uxx
    A[3, 0] = (
        1
        - (1 - mu) / r1**3
        + 3 * (1 - mu) * (x[0] + mu) ** 2 / r1**5
        - mu / r2**3
        + 3 * mu * (x[0] - 1 + mu) ** 2 / r2**5
    )
    # Uyy
    A[4, 1] = (
        1
        - (1 - mu) / r1**3
        + 3 * (1 - mu) * x[1] ** 2 / r1**5
        - mu / r2**3
        + 3 * mu * x[1] ** 2 / r2**5
    )
    # Uzz
    A[5, 2] = (
        -(1 - mu) / r1**3
        + 3 * (1 - mu) * x[2] ** 2 / r1**5
        - mu / r2**3
        + 3 * mu * x[2] ** 2 / r2**5
    )
    
    # Cross terms Uxy, Uxz, Uyz
    # Uxy = 3*(1-mu)*(x+mu)*y/r1^5 + 3*mu*(x-1+mu)*y/r2^5
    Uxy = 3 * (1 - mu) * (x[0] + mu) * x[1] / r1**5 + 3 * mu * (x[0] - 1 + mu) * x[1] / r2**5
    A[3, 1] = Uxy
    A[4, 0] = Uxy
    
    # Uxz = 3*(1-mu)*(x+mu)*z/r1^5 + 3*mu*(x-1+mu)*z/r2^5
    Uxz = 3 * (1 - mu) * (x[0] + mu) * x[2] / r1**5 + 3 * mu * (x[0] - 1 + mu) * x[2] / r2**5
    A[3, 2] = Uxz
    A[5, 0] = Uxz
    
    # Uyz = 3*(1-mu)*y*z/r1^5 + 3*mu*y*z/r2^5
    Uyz = 3 * (1 - mu) * x[1] * x[2] / r1**5 + 3 * mu * x[1] * x[2] / r2**5
    A[4, 2] = Uyz
    A[5, 1] = Uyz
    
    return A

def get_stable_eigenvector(point, eig_tol=1.0e-5):
    """
    Returns the stable eigenvector for the linearized dynamics at a given point.
    Returns None if no stable eigenvalue is found.
    """
    evals, evecs = np.linalg.eig(linearized_dynamics(point))
    for i, eig in enumerate(evals):
        if eig.real < -eig_tol:
            return evecs[:, i].real
    return None
