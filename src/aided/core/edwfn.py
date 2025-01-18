"""
aided.core.EDWfn

Electron Density Manfestation abctract class.

Copyright (C) J. Robert Michael, 2025
"""

from .. import np
from .edrep import EDRep


class EDWfn(EDRep):
    """
    Electron Density Representation from a .wfn file.
    """

    def read_input_file(self, input_file: str):
        """Read the input file describing this ED which is the result of an optimization."""
        print("TODO: Do this here!")
        pass

    def read_vib_file(self, input_file: str):
        """
        Read the log file from the optimization procedure.
        Expected to include sufficient information to generate the MSDA.
        """
        raise NotImplementedError

    def read_msda_matrix(self, msda_file: str):
        """Read the MSDA matrix from a file."""

        raise NotImplementedError

    def rho(self, x: np.float32, y: np.float32, z: np.float32) -> np.float32:
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

        raise NotImplementedError

    def grad(self, x: np.float32, y: np.float32, z: np.float32) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        raise NotImplementedError

    def hess(self, x: np.float32, y: np.float32, z: np.float32) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        raise NotImplementedError
