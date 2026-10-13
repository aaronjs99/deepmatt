#!/usr/bin/env python3
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

"""
Main execution script for DeepMAtt.

Usage:
    python run.py --mode <mode_name>

Example:
    python run.py --mode formation
    python run.py --mode generate_results
"""

from pathlib import Path
import sys


# Allow the documented launcher to work directly from a source checkout.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deepmatt.cli import main

if __name__ == "__main__":
    main()
