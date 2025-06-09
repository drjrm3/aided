"""
Read in an MSDA to a file.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from pathlib import Path
from aided import np, npt

from aided import get_logger

logger = get_logger()


def read_msda(filename: str) -> npt.NDArray:
    """Reads in an MSDA from a file.

    Args:
        filename (str): The name of the msda file.

    Returns:
        msda (np.ndarray): The msda as a 2D numpy array.
    """
    logger.debug(f"Reading in MSDA from {filename}")

    if not Path(filename).is_file():
        raise FileNotFoundError(f"File {filename} does not exist.")

    with open(filename, "r") as f:
        lines = f.readlines()

    n_atoms = (len(lines[0].strip().split()) - 1) // 3
    logger.debug(f"Number of atoms: {n_atoms}")

    if len(lines) != n_atoms * 3:
        raise ValueError(f"File {filename} does not have the correct number of lines.")

    msda = np.zeros((n_atoms * 3, n_atoms * 3))

    for i, line in enumerate(lines):
        parts = line.strip().split()
        msda[i, :] = [float(x) for x in parts[1:]]

    return msda
