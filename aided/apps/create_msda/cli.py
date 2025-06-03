"""Arguments specific to the command line interface.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from argparse import ArgumentParser


def add_arguments(parser: ArgumentParser) -> None:
    """Parse command line arguments.

    Args:
        parser (ArgumentParser): The argument parser instance.
    """

    parser.add_argument("-l", "--log-file", type=str, required=True, help="WFN log file")
    parser.add_argument("-o", "--output-file", type=str, required=True, help="Output MSDA file")
    parser.add_argument(
        "-T", type=float, default=300.0, help="Temperature in Kelvin (default: 300.0)"
    )
