"""
cli.py

Commandline utilities and main method call.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import argparse
import sys
from typing import List

from .version import __version__


def parse_args(argv: List[str]):
    """Parse arguments from commandline program."""

    parser = argparse.ArgumentParser("aided")
    parser.add_argument("-c", "--config", type=str, help="JSON config file.")
    parser.add_argument("-v", "--version", action="version", version=f"aided {__version__}")

    args = parser.parse_args(argv)
    return args


def main():  # pragma: no cover
    _args = parse_args(sys.argv[1:])
    print("Inside main")

    return 0


if __name__ == "__main__":  # pragma: no cover
    main()
