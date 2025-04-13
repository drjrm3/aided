"""
aided.core.EDWfn

Electron Density Manfestation abstract class.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from numpy.typing import NDArray

from .edrep import EDRep
from .. import LMNS, np
from ..io.read_wfn import read_wfn_file
from aided.math._primitives import gpow
from aided.core._edwfn import gen_chi


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

    def py_gen_chi(self, x: float, y: float, z: float, ider: int) -> bool:
        """Generate the chi matrix for the given point.

        Skip this if the point is the same as the last point.

        Args:
            x, y, z: Cartesian points in global space.
            ider: Derivative order.

        Return: True if the chi matrix was generated, False otherwise.
        """

        # Skip this if the point is the same as the last point.
        if self._last_point == (x, y, z) and self._last_der == ider:
            return False
        self._last_point = (x, y, z)
        self._last_der = ider

        # Precompute constants
        nprims = self._wfn_rep.nprims
        types = self._wfn_rep.types
        centers = self._wfn_rep.centers
        atpos = self._wfn_rep.atpos
        expons = self._wfn_rep.expons

        # Extract spherical harmonics indices (l, m, n) for all primitives
        lmn = np.array([LMNS[t] for t in types])  # Shape: (nprims, 3)
        l, m, n = lmn.T

        # Extract centers and positions
        center_indices = centers
        center_coords = atpos[center_indices]  # Shape: (nprims, 3)

        # Compute coordinates relative to atomic centers
        px, py, pz = (
            x - center_coords[:, 0],
            y - center_coords[:, 1],
            z - center_coords[:, 2],
        )  # Shape: (nprims,)

        # Compute the argument of the Gaussian primitive and the exponential
        alpha = expons  # Shape: (nprims,)
        expon = np.exp(-alpha * (px**2 + py**2 + pz**2))  # Shape: (nprims,)

        # Compute powers using gpow
        xl = gpow(px, l)  # Shape: (nprims,)
        ym = gpow(py, m)  # Shape: (nprims,)
        zn = gpow(pz, n)  # Shape: (nprims,)

        # Compute `chi`
        self._chi[:nprims] = xl * ym * zn * expon  # Shape: (nprims,)

        # First derivatives (if ider >= 1)
        if ider >= 1:
            twoa = 2.0 * alpha  # Shape: (nprims,)

            term11 = gpow(px, l - 1) * l  # Shape: (nprims,)
            term12 = gpow(py, m - 1) * m  # Shape: (nprims,)
            term13 = gpow(pz, n - 1) * n  # Shape: (nprims,)

            xyexp = xl * ym * expon  # Shape: (nprims,)
            xzexp = xl * zn * expon  # Shape: (nprims,)
            yzexp = ym * zn * expon  # Shape: (nprims,)

            self._chi1[:nprims, 0] = yzexp * (term11 - twoa * xl * px)
            self._chi1[:nprims, 1] = xzexp * (term12 - twoa * ym * py)
            self._chi1[:nprims, 2] = xyexp * (term13 - twoa * zn * pz)

            # Second derivatives (if ider >= 2)
            if ider >= 2:
                twoa_chi = twoa * self._chi[:nprims]  # Shape: (nprims,)

                # xx, yy, zz
                self._chi2[:nprims, 0] = gpow(px, l - 2) * yzexp * l * (l - 1) - twoa_chi * (
                    2.0 * l + 1.0 - twoa * px**2
                )
                self._chi2[:nprims, 3] = gpow(py, m - 2) * xzexp * m * (m - 1) - twoa_chi * (
                    2.0 * m + 1.0 - twoa * py**2
                )
                self._chi2[:nprims, 5] = gpow(pz, n - 2) * xyexp * n * (n - 1) - twoa_chi * (
                    2.0 * n + 1.0 - twoa * pz**2
                )

                expee = twoa * expon  # Shape: (nprims,)
                foura_two_chi = 4.0 * alpha**2 * self._chi[:nprims]  # Shape: (nprims,)

                # xy
                self._chi2[:nprims, 1] = (
                    term11 * term12 * zn * expon
                    - term12 * xl * px * zn * expee
                    - term11 * ym * py * zn * expee
                    + px * py * foura_two_chi
                )

                # xz
                self._chi2[:nprims, 2] = (
                    term11 * term13 * ym * expon
                    - term13 * xl * px * ym * expee
                    - term11 * zn * pz * ym * expee
                    + px * pz * foura_two_chi
                )

                # yz
                self._chi2[:nprims, 4] = (
                    term12 * term13 * xl * expon
                    - term13 * ym * py * xl * expee
                    - term12 * zn * pz * xl * expee
                    + py * pz * foura_two_chi
                )
        return True

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

        rhov = float(np.sum(self._denmat * self._chi[:, np.newaxis] * self._chi[np.newaxis, :]))

        return rhov

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_chi(x, y, z, ider=1)

        gradv = np.zeros(3, dtype=float)

        # Compute pairwise products of _chi and _chi1
        chi_i_chi1_j = np.einsum("i,jk->ijk", self._chi, self._chi1)
        chi_j_chi1_i = np.einsum("j,ik->ijk", self._chi, self._chi1)

        # Combine the contributions to the gradient
        gradv = np.zeros(3, dtype=float)
        for dim in range(3):  # Iterate over x, y, z dimensions
            gradv[dim] = np.sum(self._denmat * (chi_i_chi1_j[:, :, dim] + chi_j_chi1_i[:, :, dim]))

        return gradv

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_chi(x, y, z, ider=2)

        dxx, dyy, dzz, dxy, dxz, dyz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Diagonal terms
        dxx = np.sum(
            self._denmat
            * (
                self._chi[:, np.newaxis] * self._chi2[np.newaxis, :, 0]
                + 2.0 * self._chi1[:, 0][:, np.newaxis] * self._chi1[:, 0][np.newaxis, :]
                + self._chi2[:, 0][:, np.newaxis] * self._chi[np.newaxis, :]
            )
        )
        dyy = np.sum(
            self._denmat
            * (
                self._chi[:, np.newaxis] * self._chi2[np.newaxis, :, 3]
                + 2.0 * self._chi1[:, 1][:, np.newaxis] * self._chi1[:, 1][np.newaxis, :]
                + self._chi2[:, 3][:, np.newaxis] * self._chi[np.newaxis, :]
            )
        )
        dzz = np.sum(
            self._denmat
            * (
                self._chi[:, np.newaxis] * self._chi2[np.newaxis, :, 5]
                + 2.0 * self._chi1[:, 2][:, np.newaxis] * self._chi1[:, 2][np.newaxis, :]
                + self._chi2[:, 5][:, np.newaxis] * self._chi[np.newaxis, :]
            )
        )

        # Off-diagonal terms
        dxy = np.sum(
            self._denmat
            * (
                self._chi[:, np.newaxis] * self._chi2[np.newaxis, :, 1]
                + self._chi1[:, 0][:, np.newaxis] * self._chi1[:, 1][np.newaxis, :]
                + self._chi1[:, 1][:, np.newaxis] * self._chi1[:, 0][np.newaxis, :]
                + self._chi2[:, 1][:, np.newaxis] * self._chi[np.newaxis, :]
            )
        )
        dxz = np.sum(
            self._denmat
            * (
                self._chi[:, np.newaxis] * self._chi2[np.newaxis, :, 2]
                + self._chi1[:, 0][:, np.newaxis] * self._chi1[:, 2][np.newaxis, :]
                + self._chi1[:, 2][:, np.newaxis] * self._chi1[:, 0][np.newaxis, :]
                + self._chi2[:, 2][:, np.newaxis] * self._chi[np.newaxis, :]
            )
        )
        dyz = np.sum(
            self._denmat
            * (
                self._chi[:, np.newaxis] * self._chi2[np.newaxis, :, 4]
                + self._chi1[:, 1][:, np.newaxis] * self._chi1[:, 2][np.newaxis, :]
                + self._chi1[:, 2][:, np.newaxis] * self._chi1[:, 1][np.newaxis, :]
                + self._chi2[:, 4][:, np.newaxis] * self._chi[np.newaxis, :]
            )
        )

        # Combine into Hessian vector
        hessv = np.array([dxx, dyy, dzz, dxy, dxz, dyz], dtype=float)
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
