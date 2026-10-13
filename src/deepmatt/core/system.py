"""Normalized two-primary systems used by the CR3BP models.

The equations of motion are nondimensional.  A system definition therefore
contains the secondary mass fraction together with optional characteristic
scales used only when reporting distances and times.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CR3BPSystem:
    """A normalized circular restricted three-body problem system.

    Args:
        name: Human-readable identifier for reports and plots.
        mass_parameter: Secondary-body mass fraction, strictly between zero
            and one.
        length_scale_km: Characteristic primary separation in kilometres.
        time_scale_s: Characteristic time in seconds.
    """

    name: str
    mass_parameter: float
    length_scale_km: Optional[float] = None
    time_scale_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A CR3BP system needs a non-empty name.")
        if not 0.0 < self.mass_parameter < 1.0:
            raise ValueError("mass_parameter must be strictly between 0 and 1.")
        if self.length_scale_km is not None and self.length_scale_km <= 0.0:
            raise ValueError("length_scale_km must be positive when provided.")
        if self.time_scale_s is not None and self.time_scale_s <= 0.0:
            raise ValueError("time_scale_s must be positive when provided.")

    @property
    def primary_position(self) -> tuple[float, float, float]:
        """Position of the larger primary in the rotating normalized frame."""
        return (-self.mass_parameter, 0.0, 0.0)

    @property
    def secondary_position(self) -> tuple[float, float, float]:
        """Position of the smaller primary in the rotating normalized frame."""
        return (1.0 - self.mass_parameter, 0.0, 0.0)


EARTH_MOON_SYSTEM = CR3BPSystem(
    name="Earth-Moon",
    mass_parameter=0.01215058560962404,
    length_scale_km=384400.0,
    time_scale_s=375200.0,
)
