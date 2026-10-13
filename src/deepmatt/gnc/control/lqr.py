"""Shared continuous-time feedback for acceleration-controlled states."""

import numpy as np
from scipy.linalg import solve_continuous_are


def acceleration_input_matrix() -> np.ndarray:
    """Map three control accelerations into a six-component state derivative."""
    matrix = np.zeros((6, 3))
    matrix[3:, :] = np.eye(3)
    return matrix


def lqr_gain(A: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Solve for K in u = -K x; callers own solver-failure policy.

    The control input acts on velocity derivatives. Cost matrices stay with
    each experiment because their different weights are intentional.
    """
    B = acceleration_input_matrix()
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P
