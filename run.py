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
from cislune.transfer import main as transfer_main

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
    elif args.mode == "transfer":
        transfer_main()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
