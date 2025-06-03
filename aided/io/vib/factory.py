"""
aided.io.vib.factory

Generates a log reader based on the contents of the file.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from pathlib import Path

from aided import get_logger
from .log_reader import LogReader

logger = get_logger()


def detect_log_type(logfile: str | Path) -> str:
    """Detect the type of log file based on its contents.

    Args:
        logfile (str | Path): Path to the log file.

    Returns:
        str: Type of the log file ('QChem', 'Gaussian', etc.)
    """

    logger.debug(f"Detecting log type for file: {logfile}")

    try:
        with open(logfile, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        logger.error(f"Log file not found: {logfile}")
        return ""

    for line in lines:
        if "Entering Gaussian System" in line:
            logger.debug("Detected Gaussian log file.")
            return "Gaussian"

    raise ValueError("Unsupported log file format.")


def log_reader_factory(logfile: str | Path) -> LogReader:
    """Factory function to create a log reader based on the file type.

    Args:
        logfile (str | Path): Path to the log file.

    Returns:
        LogReader: An instance of a subclass of LogReader.
    """
    log_type = detect_log_type(logfile)

    if log_type == "Gaussian":
        # pylint: disable=import-outside-toplevel
        from .gaussian import GaussianLogReader

        return GaussianLogReader(logfile)

    raise ValueError(f"Unsupported log file format: {log_type}")
