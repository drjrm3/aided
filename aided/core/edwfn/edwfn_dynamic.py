"""
aided.core.edwfn.edwfn_dynamic

Dynamic Electron Density manifestation.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from numpy.typing import NDArray
from aided import np
from aided.constants import LMNS
from aided.io.vib.reader import gen_msda
from .edwfn import EDWfn

def gaussian_product_center(A: np.ndarray, alpha: float, B: np.ndarray, beta: float) -> np.ndarray:
    """Calculate the center of a Gaussian product.

    Args:
        A: Center of the first Gaussian.
        alpha: Exponent of the first Gaussian.
        B: Center of the second Gaussian.
        beta: Exponent of the second Gaussian.

    Returns:
        C: Center of the Gaussian product.
    """
    return (alpha * A + beta * B) / (alpha + beta)

def dynamic_gaussian(X, A, B, alpha, beta, U, lmn_i, lmn_j) -> float:
    """Return the dynamic s-s orbital of a Gaussian product."""

    gamma = alpha + beta
    C = (alpha * A + beta * B) / gamma
    W = np.linalg.inv(np.eye(3)/gamma + U)

    prefactor = (gamma**-3 * np.linalg.det(W))**0.5
    Eg = np.exp(-0.5*alpha*beta/gamma * np.dot(A-B, A-B))
    expon = np.exp(-0.5 * (X-C) @ np.linalg.inv(W) @ (X-C))
    gss = prefactor * Eg * expon

    # TODO: use lmn_i, lmn_j for angular momentum considerations.

    return gss


class EDWfnDynamic(EDWfn):
    """
    Electron Density Static manifestation from a single .wfn file.
    """

    def __init__(
        self,
        wfn_file: str,
        T: float,
        log_file: str | None = None,
        msda_file: str | None = None,
        msda: NDArray[np.float64] | None = None,
    ):
        """Initialize the EDWfnDynamic object.

        Args:
            wfn_file: Path to the .wfn file.
            T: Temperature in Kelvin.
            log_file: Path to the log file.
            msda_file: Path to the MSDA file.
            msda: Precomputed MSDA array.
        """
        super().__init__(wfn_file)

        self.T = T
        self.msda = gen_msda(self.T, log_file, msda_file, msda)

    def adps_of_gaussian_pairs(self, iprim: int, jprim: int) -> np.ndarray:
        """Return the ADPs (Anisotropic Displacement Parameters) of a Gaussian product.

        Args:
            iat (int): The index of the first atom.
            jat (int): The index of the second atom.

        Returns:
            U (np.ndarray): The ADPs matrix of shape (3, 3) for the atom pair.
        """

        iat = self._wfn_rep.centers[iprim]
        jat = self._wfn_rep.centers[jprim]

        alpha = 2.0 * self._wfn_rep.centers[iat]
        beta = 2.0 * self._wfn_rep.centers[jat]

        Uaa = self.msda[3 * iat - 2 : 3 * iat, 3 * iat - 2 : 3 * iat]
        Uab = self.msda[3 * iat - 2 : 3 * iat, 3 * jat - 2 : 3 * jat]
        Uba = self.msda[3 * jat - 2 : 3 * jat, 3 * iat - 2 : 3 * iat]
        Ubb = self.msda[3 * jat - 2 : 3 * jat, 3 * jat - 2 : 3 * jat]

        U = (alpha * alpha * Uaa + alpha * beta * (Uab + Uba) + beta * beta * Ubb)
        U /= ( (alpha + beta) ** 2)

        return U

    def rho(self, x: float, y: float, z: float) -> float:
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

        X = np.array([x, y, z], dtype=np.float64)
        rhov = 0.0
        
        for iprim in range(self._nprims):
            alpha = self._wfn_rep.expons[iprim]
            lmn_i = LMNS[self._wfn_rep.types[iprim]]
            # Location of the i-th primitive Gaussian
            A = self._wfn_rep.centers[iprim]
            for jprim in range(self._nprims):
                beta = self._wfn_rep.expons[jprim]
                lmn_j = LMNS[self._wfn_rep.types[jprim]]
                B = self._wfn_rep.centers[iprim]

                U = self.adps_of_gaussian_pairs(iprim, jprim)
                gdyn = dynamic_gaussian(X, A, B, alpha, beta, U, lmn_i, lmn_j)

                rhov += self._denmat[iprim, jprim] * gdyn

        return rhov

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_gs(x, y, z, ider=1)

        # TODO: On the first implementation of generating rho, simply calculate grad as a numeric
        # Gradient calculation.

        raise NotImplementedError("Gradient calculation not implemented in EDWfnDynamic.")

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_gs(x, y, z, ider=2)

        # TODO: On the first implementation of generating rho, simply calculate hess as a numeric
        # Hessian calculation.

        raise NotImplementedError("Hessian calculation not implemented in EDWfnDynamic.")
