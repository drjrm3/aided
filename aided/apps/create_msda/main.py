"""Create a Mean Square Displace Amplitude (MSDA) file from required inputs.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from argparse import Namespace

from aided import get_logger
from aided.io.vib.factory import log_reader_factory
from aided.io.vib.writer import msda_to_file
from .cli import add_arguments  # pylint: disable=unused-import


def main(args: Namespace) -> int:
    """Main entry point."""

    logger = get_logger()

    logger.info("Creating Mean Square Displace Amplitude (MSDA) file...")
    logger.debug("Arguments: %s", args)

    reader = log_reader_factory(args.log_file)
    msda_matrix = reader.gen_msda(args.T)

    msda_to_file(msda_matrix, reader.atomic_numbers, args.output_file)

    return 0
