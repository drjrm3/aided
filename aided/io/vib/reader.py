"""
Read in an MSDA to a file.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from pathlib import Path
from aided import np, npt

from aided.io import get_logger
from .factory import log_reader_factory

logger = get_logger()


def read_msda(msda_file: str) -> npt.NDArray:
    """Reads in an MSDA from a file.

    Args:
        msda_file (str): The name of the msda file.

    Returns:
        msda (np.ndarray): The msda as a 2D numpy array.
    """
    logger.debug(f"Reading in MSDA from {msda_file}")

    if not Path(msda_file).is_file():
        raise FileNotFoundError(f"File {msda_file} does not exist.")

    with open(msda_file, "r") as f:
        lines = f.readlines()

    n_atoms = (len(lines[0].strip().split()) - 1) // 3
    logger.debug(f"Number of atoms: {n_atoms}")

    if len(lines) != n_atoms * 3:
        raise ValueError(f"File {msda_file} does not have the correct number of lines.")

    msda = np.zeros((n_atoms * 3, n_atoms * 3))

    for i, line in enumerate(lines):
        parts = line.strip().split()
        msda[i, :] = [float(x) for x in parts[1:]]

    return msda


def gen_msda(
    T: float, log_file: str | None = None, msda_file: str | None = None, msda: str | None = None
) -> npt.NDArray[np.float64]:
    """Generate the MSDA array from the log file, MSDA file, or precomputed MSDA.

    Args:
        T: Temperature in Kelvin.
        log_file: Path to the log file.
        msda_file: Path to the MSDA file.
        msda: Precomputed MSDA array.

    Returns:
        The MSDA array as a NumPy array.
    """

    # If msda is provided, return it as a NumPy array.
    if msda is not None:
        return np.array(msda, dtype=np.float64)

    # If msda_file is provided, read the MSDA from the file.
    if msda_file is not None:
        return read_msda(msda_file)

    # If log_file is provided, read the MSDA from the log file.
    if log_file is not None:
        reader = log_reader_factory(log_file)
        return reader.gen_msda(T)

    raise ValueError(
        "At least one of msda, msda_file, or log_file must be provided to generate the MSDA."
    )
