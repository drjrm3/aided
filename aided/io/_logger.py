"""
aided.io.logger

Utility functions to use for File I/O.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import logging


def get_logger(
    log_level: str = "DEBUG", log_file: str = "aided.log", initialize: bool = False
) -> logging.Logger:
    """Get a logger instance that can be used throughout the application.

    Args:
        log_level (str): The logging level to use. Defaults to "DEBUG".
        log_file (str): The file to write the log to. Defaults to "aided.log".
        initialize (bool): If True, will reinitialize the logger even if it already exists.

    Returns:
        _logger (logging.Logger): The logger instance.
    """

    log_level = log_level.upper()

    # If this logger already exists, just return it.
    _logger = logging.getLogger("aided")
    if not initialize:
        return _logger

    # Otherwise, configure it.
    _logger.setLevel(log_level)

    # Create a file handler for writing the log file
    fh_fmt = "%(asctime)s [%(levelname)s] [%(module)s.%(filename)s:%(lineno)d] %(message)s"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fh_fmt))

    # Create a console handler too - use the log_level specified.
    ch = logging.StreamHandler()
    ch.setLevel(log_level)

    # Create a log formatter that includes the filename:
    ch_fmt = "%(asctime)s [%(levelname)s] [%(name)s][%(module)s] %(message)s"
    if log_level == "DEBUG":
        ch_fmt = (
            "%(asctime)s [%(levelname)s] [%(name)s][%(module)s.%(filename)s:%(lineno)d] %(message)s"
        )
    ch.setFormatter(logging.Formatter(ch_fmt))

    # Add the handlers to the logger
    _logger.addHandler(fh)
    _logger.addHandler(ch)

    return _logger
