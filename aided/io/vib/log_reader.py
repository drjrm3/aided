"""
aided.io.vib.log_reader

Base class for log readers.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

from aided import np, npt


class LogReader(ABC):
    """Base class for log readers."""

    def __init__(self, logfile: str | Path):
        self.logfile = Path(logfile)
        self.lines = self._read_file()

        # Information about molecular structure.
        self.atomic_numbers: List[int] = []
        self.coordinates: List[Tuple[float, float, float]] = []

    def _read_file(self):
        with self.logfile.open("r", encoding="utf-8") as file:
            return file.readlines()

    @property
    def natoms(self) -> int:
        return len(self.atomic_numbers)

    @property
    @abstractmethod
    def nmodes(self) -> int:  # pragma: no cover
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def gen_msda(self, temperature: float) -> npt.NDArray[np.float64]:  # pragma: no cover
        """Generate the mass-weighted displacement array."""
        raise NotImplementedError("Subclasses must implement this method.")
