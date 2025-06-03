"""
Write out MSDA to a file.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import List
from aided import npt
from aided.constants import ATOMIC_NAMES


def msda_to_file(msda: npt.NDArray, atomic_numbers: List[int], filename: str) -> None:
    """Write the MSDA data structure to a file.

    Args:
        msda (npt.NDArray): The MSDA data structure to write.
        atomic_numbers (List[int]): List of atomic numbers corresponding to the MSDA data.
        filename (str): The name of the file to write to.
    """

    atnames = [ATOMIC_NAMES[Z] for Z in atomic_numbers]

    if msda.ndim != 2:
        raise ValueError("msda must be a 2D array.")

    if msda.shape[0] % 3 != 0:
        raise ValueError("The number of rows in msda must be a multiple of 3.")

    if msda.shape[0] != msda.shape[1]:
        raise ValueError("msda must be a square matrix.")

    if len(atnames) != msda.shape[0] // 3:
        raise ValueError("Length of atomic_numbers must be one-third the number of rows in msda.")

    with open(filename, "w", encoding="utf-8") as fout:
        for irow, row in enumerate(msda):

            xyz = {0: "x", 1: "y", 2: "z"}.get(irow % 3)
            iat = irow // 3
            atname = atnames[iat]

            values = " ".join(f"{x:+20.15e}" for x in row)
            print(f"{atname}{iat+1}{xyz}: {values}", file=fout)
