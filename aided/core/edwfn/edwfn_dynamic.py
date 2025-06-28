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

def dynamic_cartesian_prefactor(
    X: NDArray[np.float64],           # observation point  (3,)
    A: NDArray[np.float64],           # centre of primitive i (3,)
    B: NDArray[np.float64],           # centre of primitive j (3,)
    C: NDArray[np.float64],           # centre of Gaussian product (3,)
    alpha: float,                     # exponent of primitive i
    beta:  float,                     # exponent of primitive j
    gamma: float,                     # gamma = alpha + beta
    U: NDArray[np.float_],            # (3,3) ADP / MSDA tensor
    lmn_i: NDArray[np.int_],          # (l_i, m_i, n_i)
    lmn_j: NDArray[np.int_],          # (l_j, m_j, n_j)
) -> float:
    """Cartesian *polynomial* factor multiplying the dynamic s–s kernel.

    Implements
        g_{L_i;L_j}^{dyn}(x) = g_ss^{dyn}(x) * Product(C_k(x)^{n_k})
    where the six linear factors C_1, ..., C_6 are defined in Eq. (9) of the LaTeX
    document.

    Args:
        X: Cartesian coordinates of the evaluation point.
        A: Location of primitive i.
        B: Location of primitive j.
        alpha: Gaussian exponent of primitive i.
        beta: Gaussian exponent of primitive j.
        U: Temperature-dependent anisotropic‐displacement tensor **U**.
        lmn_i: Cartesian exponents (l, m, n) on centre i.
        lmn_j: Cartesian exponents (l, m, n) on centre i.

    Returns:
        dyn_prefactor: Full Cartesian prefactor Product(C_k(x)^{n_k}) evaluated at **X**.
    """
    W_inv: NDArray = np.eye(3) / gamma + U              # G⁻¹ + U
    W_inv_xc: NDArray = W_inv @ (X - C)                 # W⁻¹ (x – c)
    AB: NDArray = A - B                                 # inter-nuclear vector

    C_i = (alpha / gamma) * W_inv_xc - (alpha * beta / gamma) * AB   # C₁..C₃
    C_j = (beta  / gamma) * W_inv_xc + (alpha * beta / gamma) * AB   # C₄..C₆

    # Raise each component to its required power -----------------------
    (l_i, m_i, n_i) = lmn_i
    (l_j, m_j, n_j) = lmn_j

    prefactor = (
        C_i[0] ** l_i * C_j[0] ** l_j *      # x-components
        C_i[1] ** m_i * C_j[1] ** m_j *      # y-components
        C_i[2] ** n_i * C_j[2] ** n_j        # z-components
    )
    return prefactor

def dynamic_gaussian(X, A, B, alpha, beta, U, lmn_i, lmn_j) -> float:
    """Return the dynamic s-s orbital of a Gaussian product."""

    gamma = alpha + beta
    C = (alpha * A + beta * B) / gamma
    W = np.linalg.inv(np.eye(3)/gamma + U)

    prefactor = (gamma**-3 * np.linalg.det(W))**0.5
    Eg = np.exp(-0.5*alpha*beta/gamma * np.dot(A-B, A-B))
    expon = np.exp(-0.5 * (X-C) @ np.linalg.inv(W) @ (X-C))
    gss = prefactor * Eg * expon

    dynamic_prefactor = dynamic_cartesian_prefactor(X, A, B, C, alpha, beta, gamma, U, lmn_i, lmn_j)

    return dynamic_prefactor * gss


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

        alpha = 2.0 * self._wfn_rep.expons[self._wfn_rep.centers[iat]]
        beta = 2.0 * self._wfn_rep.expons[self._wfn_rep.centers[jat]]

        start_i = 3 * iat
        start_j = 3 * jat

        Uaa = self.msda[start_i : start_i + 3, start_i : start_i + 3]
        Uab = self.msda[start_i : start_i + 3, start_j : start_j + 3]
        Uba = self.msda[start_j : start_j + 3, start_i : start_i + 3]
        Ubb = self.msda[start_j : start_j + 3, start_j : start_j + 3]

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
                B = self._wfn_rep.centers[jprim]

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
