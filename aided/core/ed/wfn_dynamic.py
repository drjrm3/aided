"""
aided.core.ed.wfn_dynamic

Dynamic Electron Density dynamic

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from pathlib import Path
from typing import Optional
from numpy.typing import NDArray

from aided import get_logger, np
from aided.constants import LMNS
from aided.io.vib.factory import log_reader_factory
from aided.io.vib.reader import read_msda
from .wfn_static import EDWfnStatic

from .dynamic_helpers import gss, Eg, gss_dyn

logger = get_logger()


class EDWfnDynamic(EDWfnStatic):
    """
    Electron Density WFN from a single .wfn file.

    This class generates the dynamic electron density assuming a static geometry and ground state
    wave function file is provided along with vibrational analysis.

    This inherits from the EDWfnStatic class.
    """

    def __init__(
        self,
        wfn_file: str,
        T: float,
        *,
        msda_matrix: Optional[NDArray[np.float64]] = None,
        msda_file: Optional[str | Path] = None,
        log_file: Optional[str | Path] = None,
    ):
        """Generates the dynamic Electron Density from a wfn file and *either*
            - Precalculated MSDA matrix
            - A precalculated MSDA file
            - A log file from the Gaussian vibrational analysis

        Args:
            T: Temperature in Kelvin.
            wfn_file: Path to the wfn file.
            log_file: Path to the Gaussian log file from a vibrational analysis.
            msda_matrix: Precalculated MSDA matrix.
        """

        # Initialize the parent (static) EDWfn class.
        super().__init__(wfn_file=wfn_file)

        # Temperate in Kelvin.
        self._T: float

        # Ensure that only one of log_file, msda_file, or msda_matrix is provided.
        if sum(x is not None for x in (log_file, msda_file, msda_matrix)) != 1:
            raise ValueError("Must provide exactly one of log_file, msda_file, or msda_matrix.")

        # Assign the MSDA matrix, if it was provided, or generate it from the provided input file.
        self._msda: NDArray[np.float64] = (
            msda_matrix
            if msda_matrix is not None
            else self._generate_msda(T, msda_file=msda_file, log_file=log_file)
        )

        # Assert that the MSDA matrix is the correct shape.
        if self._msda.shape != (3 * self.natoms, 3 * self.natoms):
            raise ValueError(
                f"MSDA matrix must be of shape {(3 * self.natoms, 3 * self.natoms)}, "
                f"but got {self._msda.shape}."
            )

        # s-s type densities of products of GTOS.
        self._gss: NDArray[np.float64] = np.zeros((self._nprims, self._nprims))

    @property
    def T(self) -> float:
        """Temperature in Kelvin."""
        return self._T

    @property
    def msda(self) -> NDArray[np.float64]:
        """The MSDA matrix."""
        return self._msda

    def U(self, iat: int, jat: int) -> NDArray[np.float64]:
        """The 3x3 block of the MSDA matrix for atoms iat and jat."""
        return self._msda[3 * iat : 3 * (iat + 1), 3 * jat : 3 * (jat + 1)]

    def _gen_gss(self, xyz: NDArray[np.float64]) -> NDArray[np.float64]:
        """Generate the s-s type densities of products of GTOS at a specific position.
        """

        gss = np.zeros((self._nprims, self._nprims), dtype=np.float64)

        for iprim in range(self._nprims):
            iat = self.centers[iprim]
            A = self.atpos[iat]
            alpha = self.expons[iprim]
            for jprim in range(self._nprims):
                jat = self.centers[jprim]
                B = self.atpos[jat]
                beta = self.expons[jprim]

                gamma = alpha + beta

                C = (alpha * A + beta * B) / gamma

                W = self.U(iat, jat) * np.eye(3) / gamma

                gss[iprim, jprim] = gss_dyn(xyz, A, B, C, alpha, beta, gamma, W)

        return gss

    def gss_dyn_deriv(self, x: float, y: float, z:float , D: int, d:float =1e-5) -> NDArray[np.float64]:
        """Generate the derivative of the s-s type densities of products of GTOS at a x,y,z.

        Args:
            x: Cartesian x point in global space.
            y: Cartesian y point in global space.
            z: Cartesian z point in global space.
            D: Direction of derivative (1=x, 2=y, 3=z)
            d: Step size for numerical derivatives.

        NOTE: This is just a rough first order derivative.
        """

        differential = np.array([d if D == 1 else 0,
                                 d if D == 2 else 0,
                                 d if D == 3 else 0], dtype=np.float64)
        xyz = np.array([x, y, z], dtype=np.float64)

        gss1 = self._gen_gss(xyz - differential)
        gss2 = self._gen_gss(xyz + differential)
        return (gss2 - gss1) / (2 * d)

    def _generate_msda(
        self,
        T: float,
        *,
        msda_file: Optional[str | Path] = None,
        log_file: Optional[str | Path] = None,
    ) -> NDArray[np.float64]:
        """Generates the MSDA matrix from either

        Args:
            T: Temperature in Kelvin.
            msda_file: Path to the MSDA file.
            log_file: Path to the Gaussian log file from a vibrational analysis.

        Returns:
            The MSDA matrix as a numpy array.
        """

        # If msda_file is provided, read the MSDA matrix from the file.
        if msda_file is not None:
            logger.debug(f"Reading MSDA matrix from file: {msda_file}")
            return read_msda(msda_file)
        # If log_file is provided, generate the MSDA matrix from the log file.
        if log_file is not None:
            logger.debug(f"Reading MSDA matrix from log file: {log_file}")
            return log_reader_factory(log_file).gen_msda(T)
        raise ValueError("Unexpected error in MSDA input handling.")

    def rho(self, x: float, y: float, z: float, dx: float = 1e-5) -> float:
        """Generate the ED at a point.

        Args:
            x: Cartesian x point in global space.
            y: Cartesian y point in global space.
            z: Cartesian z point in global space.
            dx: Step size for numerical derivatives.

        Returns: Value of ED in chosen units.
        """

        gss_x1 = self._gen_gss(np.array([x-dx, y, z], dtype=np.float64))
        gss_x2 = self._gen_gss(np.array([x+dx, y, z], dtype=np.float64))
        gss_y1 = self._gen_gss(np.array([x, y-dx, z], dtype=np.float64))
        gss_y2 = self._gen_gss(np.array([x, y+dx, z], dtype=np.float64))
        gss_z1 = self._gen_gss(np.array([x, y, z-dx], dtype=np.float64))
        gss_z2 = self._gen_gss(np.array([x, y, z+dx], dtype=np.float64))

        rhov = 0.0

        for iprim in range(self._nprims):
            itype = self.types[iprim]
            l1, m1, n1 = LMNS[iprim]
            for jprim in range(self._nprims):
                jtype = self.types[jprim]
                l2, m2, n2 = LMNS[jprim]

                gss_dyn = self._gss[iprim, jprim]

                irhox = 


                # Now - take the deriv


        raise NotImplementedError("rho not implemented yet")

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_gs(x, y, z, ider=1)

        raise NotImplementedError("grad not implemented yet")

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_gs(x, y, z, ider=2)

        raise NotImplementedError("hess not implemented yet")
