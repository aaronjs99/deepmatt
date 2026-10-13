"""Save complete GNC runs without overwriting earlier experiments."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import scipy

from deepmatt.gnc.integrated_sim import GNCResult


def _write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_trajectory(result, path):
    axes = ("x", "y", "z", "vx", "vy", "vz")
    state_groups = (
        ("chief", result.chief_state),
        ("reference", result.reference_state),
        ("deputy", result.deputy_state),
        ("estimated_relative", result.estimated_relative_state),
    )
    header = ["time_nd"]
    header += [f"{group}_{axis}_nd" for group, _ in state_groups for axis in axes]
    header += [
        "position_error_nd",
        "velocity_error_nd",
        "estimation_position_error_nd",
        "delta_v_nd",
    ]
    header += [
        f"{group}_{axis}_nd"
        for group in ("requested", "command", "applied")
        for axis in ("ax", "ay", "az")
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        position, velocity, estimation = (
            result.position_error,
            result.velocity_error,
            result.estimation_position_error,
        )
        for index, time in enumerate(result.time):
            row = [time] + [
                value for _, states in state_groups for value in states[index]
            ]
            row += [
                position[index],
                velocity[index],
                estimation[index],
                result.delta_v[index],
            ]
            if index < len(result.control_applied):
                row += [
                    value
                    for controls in (
                        result.control_requested,
                        result.control_command,
                        result.control_applied,
                    )
                    for value in controls[index]
                ]
            else:
                # The terminal sample has no following control interval.
                row += [""] * 9
            writer.writerow(row)


def _write_plot(result, path):
    # Use Agg directly: no interactive backend, window, or global backend change.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    system = result.config.system
    length = 1 if system.length_scale_km is None else system.length_scale_km * 1000
    position_unit = "normalized" if system.length_scale_km is None else "m"
    velocity = (
        1
        if system.length_scale_km is None or system.time_scale_s is None
        else length / system.time_scale_s
    )
    velocity_unit = (
        "normalized"
        if system.length_scale_km is None or system.time_scale_s is None
        else "m/s"
    )
    figure = Figure(figsize=(11, 7.5), facecolor="#f8fafc")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    plots = (
        (
            axes[0, 0],
            result.time,
            result.position_error * length,
            "Formation tracking",
            f"Position error ({position_unit})",
            "#2563eb",
        ),
        (
            axes[0, 1],
            result.time,
            result.estimation_position_error * length,
            "Relative navigation",
            f"Position estimation error ({position_unit})",
            "#d97706",
        ),
        (
            axes[1, 0],
            result.time,
            np.linalg.norm(result.control_applied, axis=1),
            "Applied control",
            "Acceleration (normalized)",
            "#0f766e",
        ),
        (
            axes[1, 1],
            result.time,
            result.delta_v * velocity,
            "Accumulated control usage",
            f"Delta-V ({velocity_unit})",
            "#7c3aed",
        ),
    )
    for axis, time, values, title, label, color in plots:
        if len(time) == len(values) + 1:
            axis.stairs(values, time, color=color, linewidth=1.8)
        else:
            axis.plot(time, values, color=color, linewidth=1.8)
        axis.set(title=title, xlabel="Time (normalized)", ylabel=label)
        axis.grid(alpha=0.18)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        f"GNC simulation | {system.name} | seed {result.config.seed}", fontsize=15
    )
    figure.text(
        0.5,
        0.015,
        "Configured unforced reference; simulation only. Controls held over each sample interval.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(path, dpi=150)


def save_gnc_result(result: GNCResult, output_dir: str | Path, *, plot=True) -> Path:
    """Create a unique bundle; mark complete only after every artifact is saved.

    A failed/interrupted save retains an incomplete summary and any partial
    files for diagnosis. It never replaces a prior run or reports success.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    prefix = created.strftime("%Y%m%dT%H%M%SZ") + f"-seed{result.config.seed}-"
    destination = Path(tempfile.mkdtemp(prefix=prefix, dir=output_dir))
    summary_path = destination / "summary.json"
    summary = result.summary()
    summary.update(
        {
            "status": "incomplete",
            "created_utc": created.isoformat(),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
            "artifacts": ["config.json", "trajectory.csv", "states.npz"]
            + (["performance.png"] if plot else []),
        }
    )
    _write_json(summary_path, summary)
    try:
        _write_json(destination / "config.json", result.config.to_dict())
        _write_trajectory(result, destination / "trajectory.csv")
        np.savez_compressed(
            destination / "states.npz",
            **{
                name: getattr(result, name)
                for name in (
                    "time",
                    "chief_state",
                    "reference_state",
                    "deputy_state",
                    "estimated_relative_state",
                    "covariance",
                    "control_requested",
                    "control_command",
                    "control_applied",
                    "delta_v",
                )
            },
        )
        if plot:
            _write_plot(result, destination / "performance.png")
        summary["status"] = "completed"
        _write_json(summary_path, summary)
    except BaseException as exc:
        summary["status"] = "incomplete"
        summary["error"] = str(exc) or type(exc).__name__
        try:
            _write_json(summary_path, summary)
        except OSError:
            pass
        raise
    return destination
