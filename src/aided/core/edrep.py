"""
aided.core.EDRep

Electron Density Representation abstract class.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
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
        self._units = Units.BOHR

    @property
    def units(self):
        return self._units

    @property
    def in_au(self):
        return self._units == Units.BOHR

    @abstractmethod
    def rho(self, x: float, y: float, z: float) -> float:
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

    @abstractmethod
    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

    @abstractmethod
    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """
