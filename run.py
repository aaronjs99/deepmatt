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

from deepmatt.cli import main

if __name__ == "__main__":
    main()
