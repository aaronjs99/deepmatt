"""Normalized circular restricted three-body problem dynamics.

This module deliberately contains only the numerical CR3BP model.  Optional
optimizers and legacy physical-units prototypes belong outside the reusable
dynamics API.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from deepmatt.core.system import CR3BPSystem, EARTH_MOON_SYSTEM


def cr3bp_dynamics(
    _time: float,
    state: Sequence[Any],
    *_args: Any,
    system: CR3BPSystem = EARTH_MOON_SYSTEM,
    **kwargs: Any,
) -> list[Any]:
    """Return normalized rotating-frame CR3BP state derivatives.

    Args:
        state: ``[x, y, z, vx, vy, vz]`` state in the rotating normalized
            frame.
        system: Two-primary system definition. Earth--Moon is the default.
        m: Optional math backend supplied as a keyword for symbolic callers.
    """
    if len(state) != 6:
        raise ValueError("CR3BP state must contain six position and velocity values.")

    m = kwargs.get("m", np)
    mu = system.mass_parameter
    x, y, z, vx, vy, vz = state
    r_primary = m.sqrt((x + mu) ** 2 + y**2 + z**2)
    r_secondary = m.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    return [
        vx,
        vy,
        vz,
        x + 2.0 * vy - (1.0 - mu) * (x + mu) / r_primary**3 - mu * (x - 1.0 + mu) / r_secondary**3,
        y - 2.0 * vx - (1.0 - mu) * y / r_primary**3 - mu * y / r_secondary**3,
        -(1.0 - mu) * z / r_primary**3 - mu * z / r_secondary**3,
    ]


def linearized_dynamics(
    state: Sequence[float], *, system: CR3BPSystem = EARTH_MOON_SYSTEM
) -> np.ndarray:
    """Return the 6-by-6 CR3BP state Jacobian about ``state``."""
    if len(state) != 6:
        raise ValueError("CR3BP state must contain six position and velocity values.")

    mu = system.mass_parameter
    x, y, z = np.asarray(state[:3], dtype=float)
    r_primary = np.sqrt((x + mu) ** 2 + y**2 + z**2)
    r_secondary = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2)

    jacobian = np.zeros((6, 6))
    jacobian[:3, 3:] = np.eye(3)
    jacobian[3, 4] = 2.0
    jacobian[4, 3] = -2.0

    primary_weight = 1.0 - mu
    secondary_x = x - 1.0 + mu

    jacobian[3, 0] = (
        1.0
        - primary_weight / r_primary**3
        + 3.0 * primary_weight * (x + mu) ** 2 / r_primary**5
        - mu / r_secondary**3
        + 3.0 * mu * secondary_x**2 / r_secondary**5
    )
    jacobian[4, 1] = (
        1.0
        - primary_weight / r_primary**3
        + 3.0 * primary_weight * y**2 / r_primary**5
        - mu / r_secondary**3
        + 3.0 * mu * y**2 / r_secondary**5
    )
    jacobian[5, 2] = (
        -primary_weight / r_primary**3
        + 3.0 * primary_weight * z**2 / r_primary**5
        - mu / r_secondary**3
        + 3.0 * mu * z**2 / r_secondary**5
    )

    u_xy = 3.0 * primary_weight * (x + mu) * y / r_primary**5 + 3.0 * mu * secondary_x * y / r_secondary**5
    u_xz = 3.0 * primary_weight * (x + mu) * z / r_primary**5 + 3.0 * mu * secondary_x * z / r_secondary**5
    u_yz = 3.0 * primary_weight * y * z / r_primary**5 + 3.0 * mu * y * z / r_secondary**5
    jacobian[3, 1] = jacobian[4, 0] = u_xy
    jacobian[3, 2] = jacobian[5, 0] = u_xz
    jacobian[4, 2] = jacobian[5, 1] = u_yz

    return jacobian


def get_stable_eigenvector(
    point: Sequence[float],
    eig_tol: float = 1.0e-5,
    *,
    system: CR3BPSystem = EARTH_MOON_SYSTEM,
) -> np.ndarray | None:
    """Return one stable eigenvector, or ``None`` if no stable mode exists."""
    eigenvalues, eigenvectors = np.linalg.eig(linearized_dynamics(point, system=system))
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue.real < -eig_tol:
            return eigenvectors[:, index].real
    return None
