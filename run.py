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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orbex-A run entry point. Select a mode to execute."
    )
    parser.add_argument(
        '--mode', '-m',
        default='adversary',
        help='Execution mode: adversary, halo, ...'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == 'adversary':
        # Call the adversary routine
        adversary_main()
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == '__main__':
    main()
