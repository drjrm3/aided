"""
aided.core.EDWfn

Electron Density Manfestation abstract class.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from numba import njit
from numpy.typing import NDArray

from .edrep import EDRep
from .. import np
from ..io.read_wfn import read_wfn_file
from aided.core._edwfn import gen_chi


@njit(fastmath=True, cache=True)
def numba_rho(denmat: NDArray, chi: NDArray) -> float:  # pragma: no cover
    """Compute the density using numba

    Args:
        denmat: Density matrix.
        chi: Chi matrix.

    Returns:
        Density value.
    """
    rho = 0.0
    n = chi.shape[0]
    for i in range(n):
        for j in range(n):
            rho += denmat[i, j] * chi[i] * chi[j]
    return rho


@njit(fastmath=True, cache=True)
def numba_grad(denmat: NDArray, chi: NDArray, chi1: NDArray) -> NDArray:  # pragma: no cover
    """Compute the gradient using numba

    Args:
        denmat: Density matrix.
        chi: Chi matrix.
        chi1: First derivative of chi.

    Returns:
        Gradient vector.
    """
    grad = np.zeros(3)
    n = chi.shape[0]
    for i in range(n):
        for j in range(n):
            for dim in range(3):
                grad[dim] += denmat[i, j] * (chi[i] * chi1[j, dim] + chi[j] * chi1[i, dim])
    return grad


@njit(fastmath=True, cache=True)
def numba_hess(
    denmat: NDArray, chi: NDArray, chi1: NDArray, chi2: NDArray
) -> NDArray:  # pragma: no cover
    """Compute the hessian using numba

    Args:
        denmat: Density matrix.
        chi: Chi matrix.
        chi1: First derivative of chi.
        chi2: Second derivative of chi.

    Returns:
        Hessian matrix.
    """
    hess = np.zeros(6)
    n = chi.shape[0]

    for i in range(n):
        for j in range(n):
            hess[0] += denmat[i, j] * (
                chi[i] * chi2[j, 0] + 2 * chi1[i, 0] * chi1[j, 0] + chi2[i, 0] * chi[j]
            )
            hess[1] += denmat[i, j] * (
                chi[i] * chi2[j, 3] + 2 * chi1[i, 1] * chi1[j, 1] + chi2[i, 3] * chi[j]
            )
            hess[2] += denmat[i, j] * (
                chi[i] * chi2[j, 5] + 2 * chi1[i, 2] * chi1[j, 2] + chi2[i, 5] * chi[j]
            )
            hess[3] += denmat[i, j] * (
                chi[i] * chi2[j, 1]
                + chi1[i, 0] * chi1[j, 1]
                + chi1[i, 1] * chi1[j, 0]
                + chi2[i, 1] * chi[j]
            )
            hess[4] += denmat[i, j] * (
                chi[i] * chi2[j, 2]
                + chi1[i, 0] * chi1[j, 2]
                + chi1[i, 2] * chi1[j, 0]
                + chi2[i, 2] * chi[j]
            )
            hess[5] += denmat[i, j] * (
                chi[i] * chi2[j, 4]
                + chi1[i, 1] * chi1[j, 2]
                + chi1[i, 2] * chi1[j, 1]
                + chi2[i, 4] * chi[j]
            )

    return hess


class EDWfn(EDRep):
    """
    Electron Density Representation from a single .wfn file.
    """

    def __init__(self, wfn_file: str):
        super().__init__(input_file=wfn_file)

        self._denmat: NDArray[np.float64]
        self._chi: NDArray[np.float64]
        self._chi1: NDArray[np.float64]
        self._chi2: NDArray[np.float64]
        self._occ: NDArray[np.float64]

        # Read the wfn file.
        self._wfn_rep = read_wfn_file(wfn_file)

        self._chi = np.zeros(self._wfn_rep.nprims, dtype=np.float64)
        self._chi1 = np.zeros((self._wfn_rep.nprims, 3), dtype=np.float64)
        self._chi2 = np.zeros((self._wfn_rep.nprims, 6), dtype=np.float64)
        self._denmat = np.zeros((self._wfn_rep.nprims, self._wfn_rep.nprims), dtype=float)

        # Keep track of the last point to avoid unnecessary calculations.
        self._last_point = None
        self._last_der = -1

        # Create simple abbreviations for wfn_rep. Remove this if it becomes performance bottleneck.
        self._occ = self._wfn_rep.occs
        self._mocs = self._wfn_rep.coeffs
        self._nprims = self._wfn_rep.nprims

        # Calculate the density matrix.
        self._gen_denmat()

    @property
    def atpos(self):
        return self._wfn_rep.atpos

    @property
    def atnames(self):
        return self._wfn_rep.atnames

    def _gen_chi(self, x: float, y: float, z: float, ider: int) -> bool:
        """Generate the chi matrix for the given point.

        Skip this if the point is the same as the last point.

        Args:
            x, y, z: Cartesian points in global space.
            ider: Derivative order.

        Return: True if the chi matrix was generated, False otherwise.
        """

        did_compute, self._last_point, self._last_der = gen_chi(
            x,
            y,
            z,
            ider,
            self._last_der,
            self._last_point,
            self._wfn_rep.types,
            self._wfn_rep.centers,
            self._wfn_rep.expons,
            self._wfn_rep.atpos,
            self._chi,
            self._chi1,
            self._chi2,
        )
        return did_compute

    def _gen_denmat(self):
        """Generate the density matrix for the given point.

        Denmat effectively computes:
            D_pq = sum_i occ_i * C_ip * C_iq
        """

        self._denmat = np.einsum("i,ip,iq->pq", self._occ, self._mocs, self._mocs)

    def rho(self, x: float, y: float, z: float) -> float:
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

        self._gen_chi(x, y, z, ider=0)

        rhov = numba_rho(self._denmat, self._chi)

        return rhov

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_chi(x, y, z, ider=1)

        gradv = numba_grad(self._denmat, self._chi, self._chi1)

        return gradv

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_chi(x, y, z, ider=2)

        hessv = numba_hess(self._denmat, self._chi, self._chi1, self._chi2)
        return hessv


def _tst():  # pragma: no cover
    # pylint: disable=all
    # This is a test area for validating work above.
    import argparse
    import sys

    # Get one or more input files.
    parser = argparse.ArgumentParser(description="Test wfn reading.")
    parser.add_argument("-i", "--input", type=str, nargs="+", help="Input wfn file(s)")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    edwfn = EDWfn(args.input[0])
    print(f"{edwfn.rho(0.0, 0.0, 0.0)=}")
    print(f"{edwfn.grad(0.0, 0.0, 0.0)=}")
    print(f"{edwfn.hess(0.0, 0.0, 0.0)=}")
    bcp = edwfn.bcp(0, 0, 0)
    print(f"{bcp=}")

    surfaces = []
    for atname in edwfn.atnames:
        print(atname)
        thetas, phis, surface = edwfn.bader_surface_of_atom(atom_name=atname, ntheta=3, nphi=20)
        surfaces.append(surface)


if __name__ == "__main__":  # pragma: no cover
    _tst()
