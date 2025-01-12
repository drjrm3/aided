"""
cli.py

Commandline utilities and main method call.

Copyright (C) J. Robert Michael, 2025
"""

import argparse
import sys
from typing import List


def parse_args(argv: List[str]):
    """Parse arguments from commandline program."""

    parser = argparse.ArgumentParser("aided")
    parser.add_argument("-c", "--config", type=str, help="JSON config file.")

    args = parser.parse_args()
    return args


def main():
    print("Inside main")

    args = parse_args(sys.argv[1:])

    return


if __name__ == "__main__":
    main()
