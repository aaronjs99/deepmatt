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

import yaml
import numpy as np
from typing import Tuple
from sympy import symbols, Eq, solve


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing the configuration parameters.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = derive_values(config)
    return config


# add a function to derive values from the config
def derive_values(config: dict) -> dict:
    """
    Derive values based on the configuration parameters.

    Args:
        config: Dictionary containing the configuration parameters.

    Returns:
        Dictionary with derived values.
    """
    if "mu_M" not in config or "mu_E" not in config:
        raise ValueError("Configuration must contain 'mu_M' and 'mu_E' parameters.")

    config["derived"] = {}
    config["derived"]["mu"] = config["mu_M"] / (config["mu_E"] + config["mu_M"])
    config["derived"]["p_E"] = (-config["derived"]["mu"], 0.0, 0.0)
    config["derived"]["p_M"] = (1 - config["derived"]["mu"], 0.0, 0.0)

    L1, L2 = get_lagrange_points(config["derived"]["mu"])
    config["derived"]["L1"] = L1
    config["derived"]["L2"] = L2

    return config


def get_lagrange_points(mu: float) -> Tuple[float, float]:
    """
    Calculate the Lagrange points L1 and L2 for a two-body system.

    Args:
        mu: Mass ratio of the secondary body to the total mass.

    Returns:
        Tuple containing the positions of L1 and L2.
    """
    if not (0 < mu < 1):
        raise ValueError("mu must be in the range (0, 1)")

    # Define the equations for L1 and L2
    x = symbols("x")
    eqL1 = Eq(
        x**5
        + ((mu - 3)) * x**4
        + (3 - 2 * mu) * x**3
        + (-mu) * x**2
        + ((2 * mu)) * x
        + (-mu),
        0,
    )
    eqL2 = Eq(
        x**5
        + (-(mu - 3)) * x**4
        + (3 - 2 * mu) * x**3
        + (-mu) * x**2
        + (-(2 * mu)) * x
        + (-mu),
        0,
    )
    L1 = (1 - mu) - solve(eqL1, dict=True)[0][x]
    L2 = (1 - mu) + solve(eqL2, dict=True)[0][x]

    return L1, L2


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


if __name__ == "__main__":
    L1, L2 = get_lagrange_points(0.1)
    print(f"L1: {L1}, L2: {L2}")  # Example usage to get Lagrange points