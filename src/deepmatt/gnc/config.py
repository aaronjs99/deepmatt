"""Validated settings for the normalized GNC experiment.

All state, time, noise, and acceleration inputs are nondimensional CR3BP
quantities. Reporting scales never change the equations of motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Mapping

import yaml

from deepmatt.core.params import L1_HALO_INITIAL_STATE
from deepmatt.core.system import CR3BPSystem, EARTH_MOON_SYSTEM


def _number(value, name, *, minimum=0.0, inclusive=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if isinstance(value, bool) or not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    if minimum is not None and (result < minimum if inclusive else result <= minimum):
        comparison = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be {comparison} {minimum}.")
    return result


def _state(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError(f"{name} must contain six values: x, y, z, vx, vy, vz.")
    return tuple(_number(item, name, minimum=None) for item in value)


@dataclass(frozen=True)
class GNCConfig:
    """Resolved inputs; Q is a discrete covariance per navigation prediction."""

    duration: float = 0.3
    time_step: float = 0.01
    seed: int = 7
    system: CR3BPSystem = EARTH_MOON_SYSTEM
    chief_initial_state: tuple[float, ...] = tuple(L1_HALO_INITIAL_STATE)
    relative_offset: tuple[float, ...] = (2.6e-5, 0, 0, 0, 0, 0)
    initial_relative_error: tuple[float, ...] = (1e-6, -1e-6, 0, 0, 0, 0)
    initial_estimate_std: float = 1e-8
    measurement_std: float = 1e-8
    process_noise_variance: float = 1e-18
    thruster_noise_std: float = 1e-5
    thruster_bias: float = 1e-6
    thruster_deadband: float = 1e-8
    max_acceleration: float = 0.05
    lqr_state_weights: tuple[float, ...] = (1e4, 1e4, 1e4, 1e2, 1e2, 1e2)
    lqr_control_weight: float = 0.1
    rtol: float = 1e-9
    atol: float = 1e-12

    def __post_init__(self):
        positive = (
            "duration",
            "time_step",
            "initial_estimate_std",
            "measurement_std",
            "max_acceleration",
            "lqr_control_weight",
            "rtol",
            "atol",
        )
        nonnegative = (
            "process_noise_variance",
            "thruster_noise_std",
            "thruster_deadband",
        )
        for name in positive + nonnegative:
            object.__setattr__(
                self,
                name,
                _number(
                    getattr(self, name),
                    name,
                    inclusive=name in nonnegative,
                ),
            )
        object.__setattr__(
            self,
            "thruster_bias",
            _number(
                self.thruster_bias,
                "thruster_bias",
                minimum=None,
            ),
        )
        for name in ("initial_estimate_std", "measurement_std"):
            deviation = getattr(self, name)
            _number(deviation * deviation, f"{name} squared (covariance)")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer.")
        if self.duration / self.time_step > 100_000:
            raise ValueError(
                "A run is limited to 100000 intervals; increase time_step or reduce duration."
            )
        if self.rtol < 2.3e-14:
            raise ValueError("rtol must be at least 2.3e-14 for double precision.")
        for name in (
            "chief_initial_state",
            "relative_offset",
            "initial_relative_error",
        ):
            object.__setattr__(self, name, _state(getattr(self, name), name))
        weights = _state(self.lqr_state_weights, "lqr_state_weights")
        if any(weight <= 0 for weight in weights):
            raise ValueError("lqr_state_weights must all be positive.")
        object.__setattr__(self, "lqr_state_weights", weights)
        if not isinstance(self.system, CR3BPSystem):
            raise ValueError("system must be a CR3BPSystem.")
        for name in ("length_scale_km", "time_scale_s"):
            value = getattr(self.system, name)
            if value is not None:
                _number(value, f"system.{name}")
        for offset in (
            (0,) * 6,
            self.relative_offset,
            tuple(
                a + b for a, b in zip(self.relative_offset, self.initial_relative_error)
            ),
        ):
            position = tuple(
                a + b for a, b in zip(self.chief_initial_state[:3], offset[:3])
            )
            if not all(math.isfinite(value) for value in position):
                raise ValueError("Initial positions must be finite.")
            if (
                min(
                    math.dist(position, primary)
                    for primary in (
                        self.system.primary_position,
                        self.system.secondary_position,
                    )
                )
                < 1e-9
            ):
                raise ValueError("An initial state is at a CR3BP primary singularity.")

    def to_dict(self):
        """JSON/YAML-compatible settings, including the full system definition."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping):
        if not isinstance(values, Mapping):
            raise ValueError("GNC configuration must be a mapping of named settings.")
        values = dict(values)
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown GNC settings: {', '.join(map(str, unknown))}")
        if "system" in values:
            system = values["system"]
            if not isinstance(system, Mapping):
                raise ValueError("system must be a mapping of CR3BP system settings.")
            try:
                system = dict(system)
                for name in ("mass_parameter", "length_scale_km", "time_scale_s"):
                    if name in system and system[name] is not None:
                        system[name] = _number(system[name], f"system.{name}")
                values["system"] = CR3BPSystem(**system)
            except (TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid system settings: {exc}") from exc
        return cls(**values)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently changing a run's settings."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("Configuration keys must be strings.")
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def load_gnc_config(path: str | Path) -> GNCConfig:
    """Read safe YAML or a saved config.json; reject malformed/unknown settings."""
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            values = yaml.load(stream, Loader=_UniqueKeyLoader)
        return GNCConfig.from_mapping(values)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Invalid GNC configuration {path}: {exc}") from exc
