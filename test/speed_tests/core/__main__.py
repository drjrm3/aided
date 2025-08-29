"""
Main routine for testing speed of various operations.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import argparse

from .edwfn_test import main as edwfn
from .math_test import main as math


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run speed tests.")
    parser.add_argument(
        "--test",
        type=str,
        choices=["edwfn", "math"],
        help="Specify the test to run.",
        required=True,
    )
    parser.add_argument("--wfnfile", type=str, help="WFN file to use for speed test.")
    parser.add_argument(
        "--num_iters", type=int, help="Number of iterations for tests.", default=1000
    )

    return parser.parse_args()


def main():
    """Main function to run the specified test."""
    args = parse_args()
    if args.test == "edwfn":
        edwfn(args)
    elif args.test == "math":
        math(args)


main()
