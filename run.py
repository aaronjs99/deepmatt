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

import argparse
import sys

# Import your modules here
from cislune.adversary import main as adversary_main
from cislune.controlled_halo import main as halo_main
from cislune.lissajous import main as lissajous_main
from cislune.stability import main as stability_main
from cislune.uncontrolled_halo import main as uncontrolled_halo_main
from cislune.stationkeeping import main as stationkeeping_main
from cislune.transfer import main as transfer_main
from cislune.poincare import main as poincare_main
from cislune.formation import main as formation_main

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orbex-A run entry point. Select a mode to execute."
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="adversary",
        help="Execution mode: adversary, halo, uncontrolled_halo, lissajous, stability, stationkeeping, transfer",
    )
    return parser.parse_args()


def main():
    args = parse_args()

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
    elif args.mode == "transfer_earth":
        from cislune.transfer import find_transfer
        find_transfer()
    elif args.mode == "transfer_leo":
        from cislune.transfer import optimize_leo_transfer
        optimize_leo_transfer()
    elif args.mode == "transfer_fast":
        from cislune.transfer import optimize_time_transfer
        optimize_time_transfer()
    elif args.mode == "poincare":
        poincare_main()
    elif args.mode == "formation":
        formation_main()
    elif args.mode == "generate_results":
        print("Generating all results...")
        uncontrolled_halo_main()
        # controlled_halo_main() # Skip for now as it uses different solver/might be slow
        lissajous_main()
        stability_main()
        
        from cislune.transfer import find_transfer, optimize_leo_transfer, optimize_time_transfer
        find_transfer()
        optimize_leo_transfer()
        optimize_time_transfer()
        
        poincare_main()
        formation_main()
        print("All results generated in results/")
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
