"""
"""

import argparse
import sys
from typing import List

def get_args(argv: List[str]):
    """
    TODO:
    """

    parser = argparse.ArgumentParser("aided")

    parser.add_argument("-w", "--wfn-files", nargs="+", type=str, help="WFN(s)")
    parser.add_argument("-c", "--config-file", type=str, help="JSON config file.")


def main():
    args = get_args(sys.argv[1:])


if __name__ == "__main__":
    main()