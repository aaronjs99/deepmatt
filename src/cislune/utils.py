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


class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def copy(self):
        return Vector(self.x, self.y, self.z)

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)


def integrate_simulator(
    sys,
    p_0: Vector,
    v_0: Vector = Vector(0.0, 0.0, 0.0),
    t_0=0.0,
    dt=0.01,
    num_steps=1000,
    *args,
    **kwargs,
):
    pos = p_0.copy()
    vel = v_0.copy()
    list_of_pos = []
    list_of_vel = []

    for time_step in range(num_steps):
        time = t_0 + time_step * dt
        acc = sys(time, pos, vel, *args, **kwargs)

        vel.x += acc.x * dt
        vel.y += acc.y * dt
        vel.z += acc.z * dt

        pos.x += vel.x * dt
        pos.y += vel.y * dt
        pos.z += vel.z * dt

        list_of_pos.append(Vector(pos.x, pos.y, pos.z))
        list_of_vel.append(Vector(vel.x, vel.y, vel.z))

    return list_of_pos, list_of_vel


def cn(n, gamma, L_type, mu):
    if L_type == 1:
        sign = 1
    elif L_type == 2:
        sign = -1
    else:
        raise ValueError("Invalid L_type")
    return (
        ((sign) ** n) * mu
        + ((-1) ** n)
        * (((1 - mu) * (gamma ** (n + 1))) / ((1 - (sign * gamma)) ** (n + 1)))
    ) / (gamma**3)


def eval_poly1d(coefficients, x):
    result = 0
    for i, coef in enumerate(coefficients[::-1]):
        if coef != 0:
            result += coef * (x**i)
    return result
