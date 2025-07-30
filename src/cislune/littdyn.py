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

from cislune.utils import *
from cislune.params import *


def gekko_simulator(
    p_0: Vector, v_0: Vector = Vector(0.0, 0.0, 0.0), dt=1, num_steps=1000
):
    m = GEKKO(remote=True)
    m.time = np.linspace(0, dt * num_steps, num_steps)

    p = [m.Var(fixed_initial=True) for i in range(3)]
    v = [m.Var(fixed_initial=True) for i in range(3)]
    u = [0 for i in range(3)]
    p[0].value = p_0.x
    p[1].value = p_0.y
    p[2].value = p_0.z
    v[0].value = v_0.x
    v[1].value = v_0.y
    v[2].value = v_0.z
    # M  = m.Var  ()

    # ux = m.Var()
    # uy = m.Var()
    # uz = m.Var()
    # T  = m.Intermediate(M*m.sqrt(ux**2 + uy**2 + uz**2))

    # U = m.Var()

    eqs = []

    a = acc(p, v, u, m=m)
    for i in range(3):
        eqs.append(p[i].dt() == v[i])
        eqs.append(v[i].dt() == a[i])
    # eqs.append(vx.dt() == ux + 2*vy + px - (1-mu)*(px+mu)/(m.sqrt((px-p_E.x)**2 + (py-p_E.y)**2 + (pz-p_E.z)**2))**3 - mu*(px-(1-mu))/(m.sqrt((px-p_M.x)**2 + (py-p_M.y)**2 + (pz-p_M.z)**2))**3)
    # eqs.append(vy.dt() == uy - 2*vx + py - (1-mu)* py    /(m.sqrt((px-p_E.x)**2 + (py-p_E.y)**2 + (pz-p_E.z)**2))**3 - mu* py        /(m.sqrt((px-p_M.x)**2 + (py-p_M.y)**2 + (pz-p_M.z)**2))**3)
    # eqs.append(vz.dt() == uz +           - (1-mu)* pz    /(m.sqrt((px-p_E.x)**2 + (py-p_E.y)**2 + (pz-p_E.z)**2))**3 - mu* pz        /(m.sqrt((px-p_M.x)**2 + (py-p_M.y)**2 + (pz-p_M.z)**2))**3)
    # eqs.append(M .dt() == -T/(Isp*g_0))

    # eqs.append(U == 0.5*(px**2 + py**2) + mu/(m.sqrt((px-p_E.x)**2 + (py-p_E.y)**2 + (pz-p_E.z)**2)) + (1-mu)/(m.sqrt((px-p_M.x)**2 + (py-p_M.y)**2 + (pz-p_M.z)**2)))

    m.Equations(eqs)

    m.options.IMODE = 4
    m.options.SOLVER = 1
    m.options.MAX_ITER = 1000

    integrator = m.Param(value=np.ones(num_steps))
    m.Minimize(integrator * (np.sum([v[i] ** 2 for i in range(3)])))

    m.solve(disp=True)
    # m.cleanup()
    # del m

    return [Vector(p[0][i], p[1][i], p[2][i]) for i in range(num_steps)]


def acc(t, p: Union[list, Vector], v: Union[list, Vector], *args, **kwargs):
    if isinstance(p, list):
        m = kwargs["m"]
        a_X = (
            2 * v[1]
            + p[0]
            - (1 - mu)
            * (p[0] + mu)
            / (
                m.sqrt((p[0] - p_E.x) ** 2 + (p[1] - p_E.y) ** 2 + (p[2] - p_E.z) ** 2)
                ** 3
            )
            - mu
            * (p[0] - (1 - mu))
            / (
                m.sqrt((p[0] - p_M.x) ** 2 + (p[1] - p_M.y) ** 2 + (p[2] - p_M.z) ** 2)
                ** 3
            )
        )
        a_Y = (
            -2 * v[0]
            + p[1]
            - (1 - mu)
            * p[1]
            / (
                m.sqrt((p[0] - p_E.x) ** 2 + (p[1] - p_E.y) ** 2 + (p[2] - p_E.z) ** 2)
                ** 3
            )
            - mu
            * p[1]
            / (
                m.sqrt((p[0] - p_M.x) ** 2 + (p[1] - p_M.y) ** 2 + (p[2] - p_M.z) ** 2)
                ** 3
            )
        )
        a_Z = -(1 - mu) * p[2] / (
            m.sqrt((p[0] - p_E.x) ** 2 + (p[1] - p_E.y) ** 2 + (p[2] - p_E.z) ** 2) ** 3
        ) - mu * p[2] / (
            m.sqrt((p[0] - p_M.x) ** 2 + (p[1] - p_M.y) ** 2 + (p[2] - p_M.z) ** 2) ** 3
        )
        if "u" in kwargs:
            a_X += kwargs["u"][0]
            a_Y += kwargs["u"][1]
            a_Z += kwargs["u"][2]
        return (a_X, a_Y, a_Z)
    elif isinstance(p, Vector):
        a_X = (
            2 * v.y
            + p.x
            - (1 - mu)
            * (p.x + mu)
            / (
                np.sqrt((p.x - p_E.x) ** 2 + (p.y - p_E.y) ** 2 + (p.z - p_E.z) ** 2)
                ** 3
            )
            - mu
            * (p.x - (1 - mu))
            / (
                np.sqrt((p.x - p_M.x) ** 2 + (p.y - p_M.y) ** 2 + (p.z - p_M.z) ** 2)
                ** 3
            )
        )
        a_Y = (
            -2 * v.x
            + p.y
            - (1 - mu)
            * p.y
            / (
                np.sqrt((p.x - p_E.x) ** 2 + (p.y - p_E.y) ** 2 + (p.z - p_E.z) ** 2)
                ** 3
            )
            - mu
            * p.y
            / (
                np.sqrt((p.x - p_M.x) ** 2 + (p.y - p_M.y) ** 2 + (p.z - p_M.z) ** 2)
                ** 3
            )
        )
        a_Z = -(1 - mu) * p.z / (
            np.sqrt((p.x - p_E.x) ** 2 + (p.y - p_E.y) ** 2 + (p.z - p_E.z) ** 2) ** 3
        ) - mu * p.z / (
            np.sqrt((p.x - p_M.x) ** 2 + (p.y - p_M.y) ** 2 + (p.z - p_M.z) ** 2) ** 3
        )
        if "u" in kwargs:
            a_X += kwargs["u"].x
            a_Y += kwargs["u"].y
            a_Z += kwargs["u"].z
        return Vector(a_X, a_Y, a_Z)
    else:
        raise ValueError("Invalid type for p")


if __name__ == "__main__":
    p_0 = Vector(0.00, 0.01, 0.01)
    v_0 = Vector(0.00, 0.00, 0.00)
    dt = 0.001
    num_steps = int(10000)

    positions = integrate_simulator(
        sys=acc, p_0=p_0, v_0=v_0, t_0=0.0, dt=dt, num_steps=num_steps
    )

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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
    axes[0].set_ylabel("X")

    axes[1].plot(np.arange(0, dt * num_steps, dt), y_coords)
    axes[1].set_ylabel("Y")

    axes[2].plot(np.arange(0, dt * num_steps, dt), z_coords)
    axes[2].set_ylabel("Z")
    axes[2].set_xlabel("t")

    plt.show()
