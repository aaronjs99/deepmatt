# /***********************************************************
# *                                                         *
# * Copyright (c) 2025                                      *
# *                                                         *
# * University of California, Los Angeles                   *
# *                                                         *
# * Author : Aaron John-Sabu                                *
# * Contact: aaronjs@g.ucla.edu                             *
# *                                                         *
# ***********************************************************/

import argparse
import sys

"""
DeepMAtt Command Line Interface.

This module provides the main entry point for the DeepMAtt framework, dispatching
commands to various subsystems including:
- Dynamics analysis (Halo, Lissajous, Stability)
- Trajectory optimization (Transfers)
- Formation flying simulation
- Deep learning training (PPO, Transformer)
- Validation and benchmarking
"""

# Analysis Modules
from deepmatt.analysis.adversary import main as adversary_main
from deepmatt.analysis.controlled_halo import main as halo_main
from deepmatt.analysis.lissajous import main as lissajous_main
from deepmatt.analysis.stability import main as stability_main
from deepmatt.analysis.uncontrolled_halo import main as uncontrolled_halo_main
from deepmatt.analysis.stationkeeping import main as stationkeeping_main
from deepmatt.analysis.transfer import (
    main as transfer_main,
    find_transfer,
    optimize_leo_transfer,
    optimize_time_transfer,
)
from deepmatt.analysis.poincare import main as poincare_main

# Formation Modules
from deepmatt.formation.control import main as formation_main
from deepmatt.formation.rigid import main as rigid_formation_main
from deepmatt.formation.optimize import optimize_formation

# Learning Modules
from deepmatt.learning.ppo_agent import train_ppo
from deepmatt.learning.transformer_guidance import train_transformer
from deepmatt.validation.benchmark_hybrid import run_benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepMAtt: Deep Manifold Attention for Cislunar Astrodynamics"
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="adversary",
        help="Execution mode (e.g., halo, stability, train_ppo, benchmark_hybrid)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Dynamics & Analysis ---
    if args.mode == "adversary":
        adversary_main()
    elif args.mode == "halo":
        halo_main()
    elif args.mode == "uncontrolled_halo":
        uncontrolled_halo_main()
    elif args.mode == "lissajous":
        lissajous_main()
    elif args.mode == "stability":
        stability_main()
    elif args.mode == "stationkeeping":
        stationkeeping_main()
    elif args.mode == "poincare":
        poincare_main()

    # --- Transfers ---
    elif args.mode == "transfer_earth":
        find_transfer()
    elif args.mode == "transfer_leo":
        optimize_leo_transfer()
    elif args.mode == "transfer_fast":
        optimize_time_transfer()

    # --- Formation Flying ---
    elif args.mode == "formation":
        formation_main()
    elif args.mode == "formation_rigid":
        rigid_formation_main()
    elif args.mode == "formation_optimize":
        optimize_formation()

    # --- Learning-Based GNC ---
    elif args.mode == "train_ppo":
        train_ppo()
    elif args.mode == "train_transformer":
        train_transformer()
    elif args.mode == "benchmark_hybrid":
        run_benchmark()

    # --- Batch Execution ---
    elif args.mode == "generate_results":
        print("Generating all results...")
        uncontrolled_halo_main()
        lissajous_main()
        stability_main()

        find_transfer()
        optimize_leo_transfer()
        optimize_time_transfer()

        poincare_main()
        formation_main()
        rigid_formation_main()
        optimize_formation()

        run_benchmark()  # Include hybrid benchmark

        print("All results generated in results/")

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
