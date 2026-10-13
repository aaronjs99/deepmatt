"""Command-line entry point for DeepMAtt experiments.

Commands are resolved only when selected. This keeps the core package usable
on machines that do not install the optional learning or optimization stacks.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from collections.abc import Callable


CommandTarget = tuple[str, str]

_COMMANDS: dict[str, CommandTarget] = {
    "gnc": ("deepmatt.gnc.integrated_sim", "integrated_gnc_sim"),
    "adversary": ("deepmatt.analysis.adversary", "main"),
    "halo": ("deepmatt.analysis.controlled_halo", "main"),
    "uncontrolled_halo": ("deepmatt.analysis.uncontrolled_halo", "main"),
    "lissajous": ("deepmatt.analysis.lissajous", "main"),
    "stability": ("deepmatt.analysis.stability", "main"),
    "stationkeeping": ("deepmatt.analysis.stationkeeping", "main"),
    "poincare": ("deepmatt.analysis.poincare", "main"),
    "transfer_earth": ("deepmatt.analysis.transfer", "find_transfer"),
    "transfer_leo": ("deepmatt.analysis.transfer", "optimize_leo_transfer"),
    "transfer_fast": ("deepmatt.analysis.transfer", "optimize_time_transfer"),
    "formation": ("deepmatt.formation.control", "main"),
    "formation_rigid": ("deepmatt.formation.rigid", "main"),
    "formation_optimize": ("deepmatt.formation.optimize", "optimize_formation"),
    "train_ppo": ("deepmatt.learning.ppo_agent", "train_ppo"),
    "train_transformer": (
        "deepmatt.learning.transformer_guidance",
        "train_transformer",
    ),
    "benchmark_hybrid": ("deepmatt.validation.benchmark_hybrid", "run_benchmark"),
}

_RESULT_MODES = (
    "uncontrolled_halo",
    "lissajous",
    "stability",
    "transfer_earth",
    "transfer_leo",
    "transfer_fast",
    "poincare",
    "formation",
    "formation_rigid",
    "formation_optimize",
    "benchmark_hybrid",
)


def available_modes() -> tuple[str, ...]:
    """Return supported commands in a stable order for scripts and help text."""
    return tuple(_COMMANDS) + ("generate_results",)


def _load_command(mode: str) -> Callable[[], object]:
    """Resolve a selected command and explain a missing optional dependency."""
    module_name, function_name = _COMMANDS[mode]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise RuntimeError(
                f'Mode "{mode}" requires the optional dependency "{exc.name}". '
                "Install the applicable project extra and retry."
            ) from exc
        raise

    command = getattr(module, function_name)
    if not callable(command):
        raise TypeError(f'Mode "{mode}" resolved to a non-callable target.')
    return command


def run_mode(mode: str) -> None:
    """Run one supported mode, or the documented paper-result batch."""
    if mode == "generate_results":
        for result_mode in _RESULT_MODES:
            run_mode(result_mode)
        return
    if mode not in _COMMANDS:
        available = ", ".join(available_modes())
        raise ValueError(f'Unknown mode "{mode}". Available modes: {available}')
    _load_command(mode)()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepMAtt: cislunar formation-flying analysis and experiments"
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="adversary",
        choices=available_modes(),
        help="Experiment or analysis mode to run.",
    )
    group = parser.add_argument_group("GNC simulation (--mode gnc)")
    group.add_argument(
        "--config", type=Path, help="YAML settings or a previous run's config.json."
    )
    group.add_argument(
        "--duration", type=float, help="Simulation duration in normalized time."
    )
    group.add_argument(
        "--dt", type=float, help="Control/navigation sample step in normalized time."
    )
    group.add_argument(
        "--seed",
        type=int,
        help="Nonnegative seed for repeatable sensor/actuator noise.",
    )
    group.add_argument(
        "--output-dir",
        type=Path,
        help="Parent folder for a new, unique run bundle (default: results/runs).",
    )
    group.add_argument(
        "--no-plot",
        action="store_true",
        help="Save numerical results without rendering a plot.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse arguments without importing numerical or optional backends."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.mode != "gnc" and any(
        (
            args.config is not None,
            args.duration is not None,
            args.dt is not None,
            args.seed is not None,
            args.output_dir is not None,
            args.no_plot,
        )
    ):
        parser.error(
            "--config, --duration, --dt, --seed, --output-dir and --no-plot require --mode gnc."
        )
    return args


def _run_gnc(args) -> None:
    from dataclasses import replace

    from deepmatt.gnc.config import GNCConfig, load_gnc_config
    from deepmatt.gnc.integrated_sim import run_gnc_simulation
    from deepmatt.output.gnc import save_gnc_result

    config = GNCConfig() if args.config is None else load_gnc_config(args.config)
    overrides = {
        name: value
        for name, value in (
            ("duration", args.duration),
            ("time_step", args.dt),
            ("seed", args.seed),
        )
        if value is not None
    }
    config = replace(config, **overrides)
    print(
        f"Running GNC simulation: {config.system.name}, duration={config.duration:g}, "
        f"step={config.time_step:g}, seed={config.seed}",
        flush=True,
    )
    result = run_gnc_simulation(config)
    destination = save_gnc_result(
        result, args.output_dir or "results/runs", plot=not args.no_plot
    )
    summary = result.summary()
    if summary["final_position_error_m"] is None:
        print(
            f"Final position error: {summary['final_position_error_nd']:.6g} (normalized)"
        )
    else:
        print(f"Final position error: {summary['final_position_error_m']:.3f} m")
    if summary["total_delta_v_m_s"] is None:
        print(f"Total delta-V: {summary['total_delta_v_nd']:.6g} (normalized)")
    else:
        print(f"Total delta-V: {summary['total_delta_v_m_s']:.6g} m/s")
    print(f"GNC simulation completed. Results: {destination}")


def main(argv: list[str] | None = None) -> None:
    """Run the command selected by the command line."""
    args = parse_args(argv)
    try:
        if args.mode == "gnc":
            _run_gnc(args)
        else:
            run_mode(args.mode)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print(
            "Interrupted. No completed result is reported; an interrupted save is marked incomplete.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
