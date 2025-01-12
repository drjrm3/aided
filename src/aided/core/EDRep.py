"""
aided.core.EDRep

Electron Density Representation abstract class.

Copyright (C) J. Robert Michael, 2025
"""

from abc import ABCMeta, abstractmethod
from enum import Enum

from aided.core.units import Units

from .. import np


class EDRepType(Enum):
    WFN = 0
    RDF = 1
    NRDF = 2


class EDRep(metaclass=ABCMeta):
    """
    Electron Density Representation.

    High level class which represents the information needed to express the ED but is abstracted
    away from the type of file.

    Types of representations may include:
        - .wfn files (AIMFile)
        - .rdf files (Radial Density Function files)
        - .nrdf (Numerical RDF files)
        - .cube (???)
        - .grd (???)
        - etc.
        - etc.
    """

    def __init__(self, input_file: str):
        # EDRep type (e.g. WFN, NWchem, etc.)
        self._edrep_type = None
        self._units = Units.Bohr

    @property
    def units(self):
        return self._units

    @property
    def in_au(self):
        return self._units == Units.Bohr

    @abstractmethod
    def read_input_file(self, input_file: str):
        """Read the input file describing this ED which is the result of an optimization."""
        pass

    def read_vib_file(self, input_file: str):
        """
        Read the log file from the optimization procedure.
        Expected to include sufficient information to generate the MSDA.
        """
        pass

    def read_msda_matrix(self, msda_file: str):
        """Read the MSDA matrix from a file."""
        pass

    @abstractmethod
    def rho(self, x: np.float32, y: np.float32, z: np.float32) -> np.float32:
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """
        pass

    @abstractmethod
    def grad(self, x: np.float32, y: np.float32, z: np.float32) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """
        pass

    @abstractmethod
    def hess(self, x: np.float32, y: np.float32, z: np.float32) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """
        pass


