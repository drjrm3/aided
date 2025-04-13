#!/usr/bin/env python3

from .edwfn_test import main as edwfn
from .math_test import main as math


def parse_args():
    import argparse

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
    args = parse_args()
    if args.test == "edwfn":
        edwfn(args)
    elif args.test == "math":
        math(args)


main()
