"""
aided.core.fio.utils

Utility functions to use for File I/O.

Copyright (C) J. Robert Michael PhD, 2025
"""

import sys
from typing import List

import re

from ..core.EDRep import EDRepType


def convert_scientific_notation(data_lines):
    """
    Converts scientific notation with 'D' to 'E' in a list of strings while preserving the line structure.

    Args:
        data_lines (list of str): Each string represents a line of space-separated values.

    Returns:
        list of str: The processed data with 'D' replaced by 'E' in scientific notation, preserving line structure.
    """
    # Pattern to identify scientific notation with 'D'
    pattern = re.compile(r'([-+]?\d+\.\d+)D([+-]\d+)')

    # Replace 'D' with 'E' in scientific notation for each line
    result = [pattern.sub(r'\1E\2', line) for line in data_lines]

    return result


def is_number(s):
    """Check if a string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_edrep_type_by_file_extention(file_name: str) -> EDRepType:
    """Get the EDRep type based on file extention alone."""
    ext = file_name.split(".")[-1]
    if ext == "wfn":
        return EDRepType.WFN
    raise ValueError(f"File extension {ext} is not supported.")
