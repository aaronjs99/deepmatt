"""Abstract base class for all formation controllers."""

from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    """Abstract base class for formation control laws.

    All controllers must implement the compute_control method which
    takes the current state and returns a control input.
    """

    @abstractmethod
    def compute_control(self, state: np.ndarray, time: float) -> np.ndarray:
        """Compute control input for the given state.

        Args:
            state: Current state vector (6,) [x, y, z, vx, vy, vz]
            time: Current time (non-dimensional)

        Returns:
            Control vector u (3,) [ux, uy, uz]
        """
        pass

    def reset(self) -> None:
        """Reset controller internal state (if any).

        Called at the beginning of each simulation run.
        """
        pass
