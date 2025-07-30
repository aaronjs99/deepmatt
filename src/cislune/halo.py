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
from scipy.special import legendre
from gekko import GEKKO

from cislune.utils import load_config, Vector, cn, eval_poly1d


class HaloOrbit:
    def __init__(self, config=None):
        self.config = config if config else load_config()
        self.halo_cfg = self.config.get("halo", {})
        self.gamma = self.halo_cfg.get("gamma", 1.001090475e-2)
        self.mu = self.halo_cfg.get("mu", 0.0121505856)
        self.L_type = self.halo_cfg.get("L_type", 1)
        self.B = np.identity(3)

    def halo_accel(self, t, p, v, linearity=False, nonlinearity="full", m=None):
        gamma, L_type, mu = self.gamma, self.L_type, self.mu
        c2 = cn(2, gamma, L_type, mu)

        sqrt = m.sqrt if m else np.sqrt
        rho = sqrt(p.x**2 + p.y**2 + p.z**2)
        r1 = sqrt((p.x + mu) ** 2 + p.y**2 + p.z**2)
        r2 = sqrt((p.x - 1 + mu) ** 2 + p.y**2 + p.z**2)

        polyp = [legendre(n) for n in range(3, 12)]
        polyd = [P.deriv() for P in polyp]
        Pnlist = [eval_poly1d(P.coefficients, p.x / rho) for P in polyp]
        Pdlist = [eval_poly1d(P.coefficients, p.x / rho) for P in polyd]

        aX_lin = 2 * v.y + (2 * c2 + 1) * p.x
        aY_lin = -2 * v.x + (-c2 + 1) * p.y
        aZ_lin = -c2 * p.z

        if linearity:
            aX_nonlin = aY_nonlin = aZ_nonlin = 0.0
        else:
            if nonlinearity == "full":
                aX_lin, aY_lin, aZ_lin = 2 * v.y, -2 * v.x, 0.0
                aX_nonlin = p.x - (1 - mu) * (p.x + mu) / r1**3 - mu * (p.x - 1 + mu) / r2**3
                aY_nonlin = p.y - (1 - mu) * p.y / r1**3 - mu * p.y / r2**3
                aZ_nonlin = -(1 - mu) * p.z / r1**3 - mu * p.z / r2**3
            elif nonlinearity == "richardson":
                aX_nonlin = 1.5 * cn(3, gamma, L_type, mu) * (2*p.x**2 - p.y**2 - p.z**2) + \
                            2.0 * cn(4, gamma, L_type, mu) * p.x * (2*p.x**2 - 3*p.y**2 - 3*p.z**2)
                aY_nonlin = -3.0 * cn(3, gamma, L_type, mu) * p.x*p.y + \
                            -1.5 * cn(4, gamma, L_type, mu) * p.y * (4*p.x**2 - p.y**2 - p.z**2)
                aZ_nonlin = -3.0 * cn(3, gamma, L_type, mu) * p.x*p.z + \
                            -1.5 * cn(4, gamma, L_type, mu) * p.z * (4*p.x**2 - p.y**2 - p.z**2)
            elif nonlinearity == "poly":
                aX_nonlin = np.sum([
                    cn(n, gamma, L_type, mu) * (
                        n * rho**(n-1) * (p.x / rho) * Pnlist[n-3] +
                        rho**n * ((1 / rho) - (p.x**2 / rho**3)) * Pdlist[n-3]
                    ) for n in range(3, len(polyp) + 3)
                ])
                aY_nonlin = np.sum([
                    cn(n, gamma, L_type, mu) * (
                        n * rho**(n-1) * (p.y / rho) * Pnlist[n-3] +
                        rho**n * (-p.x * p.y / rho**3) * Pdlist[n-3]
                    ) for n in range(3, len(polyp) + 3)
                ])
                aZ_nonlin = np.sum([
                    cn(n, gamma, L_type, mu) * (
                        n * rho**(n-1) * (p.z / rho) * Pnlist[n-3] +
                        rho**n * (-p.x * p.z / rho**3) * Pdlist[n-3]
                    ) for n in range(3, len(polyp) + 3)
                ])
            else:
                aX_nonlin = aY_nonlin = aZ_nonlin = 0.0

        return [aX_lin + aX_nonlin, aY_lin + aY_nonlin, aZ_lin + aZ_nonlin]

    def run_simulation(self, p_0, v_0, t_0, dt, num_steps, timeRange, **kwargs):
        pos, vel = p_0.copy(), v_0.copy()
        list_of_pos, list_of_vel = [], []

        for time_step in range(num_steps):
            time = t_0 + time_step * dt
            acc = self.halo_accel(time, pos, vel, **kwargs)
            vel.x += acc[0] * dt
            vel.y += acc[1] * dt
            vel.z += acc[2] * dt
            pos.x += vel.x * dt
            pos.y += vel.y * dt
            pos.z += vel.z * dt
            list_of_pos.append(Vector(pos.x, pos.y, pos.z))
            list_of_vel.append(Vector(vel.x, vel.y, vel.z))

        self._plot_trajectory(list_of_pos)
        self._plot_velocity(timeRange, list_of_vel)

    def _plot_trajectory(self, positions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot([p.x for p in positions], [p.y for p in positions], [p.z for p in positions])
        ax.set_title("Halo Orbit")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    def _plot_velocity(self, time, velocities):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time, [v.x for v in velocities], label="X")
        ax.plot(time, [v.y for v in velocities], label="Y")
        ax.plot(time, [v.z for v in velocities], label="Z")
        ax.set_title("Velocity over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Velocity")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    config = load_config()
    halo = HaloOrbit(config=config)

    halo_cfg = config.get("halo", {})
    Ax = halo_cfg.get("Ax", 0.1 * 206000000 / 110000000)
    Az = halo_cfg.get("Az", 0.1 * 110000000 / 110000000)
    kappa = halo_cfg.get("kappa", 1.0)
    Ay = kappa * Ax

    p_0 = Vector(-Ax, 0, -Az)
    v_0 = Vector(0.0, Ay, 0.0)
    t_0 = 0.0
    dt = 0.01 * np.pi
    num_steps = 150
    timeRange = np.linspace(t_0, t_0 + dt * num_steps, num_steps)

    halo.run_simulation(p_0, v_0, t_0, dt, num_steps, timeRange, nonlinearity="richardson")
