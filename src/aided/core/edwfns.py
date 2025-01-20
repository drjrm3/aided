"""
aided.core.edwfns

Electron Density Representations from .wfn files.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from typing import List
from numpy.typing import NDArray

from .edrep import EDRep

from .. import LMNS, np
from ..io.read_wfn import read_wfn_files
from ..math.primitives import gpow


class EDWfns(EDRep):
    """
    Electron Density Representation from multiple .wfn file.
    """

    def __init__(self, wfn_file_list: str):
        """Initialization with list of .wfn files and number of processes to use.

        Args:
            wfn_file_list: List of .wfn files.
        """
        super().__init__(input_file=wfn_file_list)

        self._denmat: NDArray[np.float64]
        self._chi: NDArray[np.float64]
        self._chi1: NDArray[np.float64]
        self._chi2: NDArray[np.float64]
        self._occ: NDArray[np.float64]

        with open(wfn_file_list, "r") as finp:
            wfn_files = [f.strip() for f in finp.readlines()]

        # Read the wfn file.
        self._wfns_rep = read_wfn_files(wfn_files)

        # Assumes that all atnames are the same.
        self._atnames = self._wfns_rep.atnames[0]

        # Assumes that all .wfns represent the same molecule and we want the averaged position.
        self._atpos = np.mean(self._wfns_rep.atpos, axis=0)

        # Initialize the chi matrices.
        self._chi = np.zeros((self._wfns_rep.nwfns, self._wfns_rep.nprims), dtype=float)
        self._chi1 = np.zeros((self._wfns_rep.nwfns, self._wfns_rep.nprims, 3), dtype=float)
        self._chi2 = np.zeros((self._wfns_rep.nwfns, self._wfns_rep.nprims, 6), dtype=float)
        self._denmat = np.zeros((self._wfns_rep.nwfns, self._wfns_rep.nprims, self._wfns_rep.nprims), dtype=float)

        # Keep track of the last point to avoid unnecessary calculations.
        self._last_point = None
        self._last_der = -1

        # Create simple abbreviations for wfn_rep. Remove this if it becomes performance bottleneck.
        self._occ = self._wfns_rep.occs
        self._mocs = self._wfns_rep.coeffs
        self._nprims = self._wfns_rep.nprims
        self._nwfns = self._wfns_rep.nwfns

        # Calculate the density matrix.
        self._gen_denmat()

    @property
    def atpos(self):
        # Average the atomic positions.
        return self._atpos

    @property
    def atnames(self):
        # This assumes that all atnames are equal.
        return self._atnames

    def _gen_chi_worker(self, wfn_group: List[int], x: float, y: float, z: float, ider: int):
        """Worker function for generating chi matrix for the given point in parallel."""

    def _gen_chi(self, x: float, y: float, z: float, ider: int):
        """Generate the chi matrix for the given point.

        Skip this if the point is the same as the last point.

        Args:
            x, y, z: Cartesian points in global space.
            ider: Derivative order.
        """

        # Skip this if the point is the same as the last point.
        if self._last_point == (x, y, z) and self._last_der == ider:
            return
        self._last_point = (x, y, z)
        self._last_der = ider

        for iwfn in range(self._nwfns):

            # Precompute constants
            nprims = self._wfns_rep.nprims
            types = self._wfns_rep.types[iwfn]
            centers = self._wfns_rep.centers[iwfn]
            atpos = self._wfns_rep.atpos[iwfn]
            expons = self._wfns_rep.expons[iwfn]

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
            self._chi[iwfn, :nprims] = xl * ym * zn * expon  # Shape: (nprims,)

            # First derivatives (if ider >= 1)
            if ider >= 1:
                twoa = 2.0 * alpha  # Shape: (nprims,)

                term11 = gpow(px, l - 1) * l  # Shape: (nprims,)
                term12 = gpow(py, m - 1) * m  # Shape: (nprims,)
                term13 = gpow(pz, n - 1) * n  # Shape: (nprims,)

                xyexp = xl * ym * expon  # Shape: (nprims,)
                xzexp = xl * zn * expon  # Shape: (nprims,)
                yzexp = ym * zn * expon  # Shape: (nprims,)

                self._chi1[iwfn, :nprims, 0] = yzexp * (term11 - twoa * xl * px)
                self._chi1[iwfn, :nprims, 1] = xzexp * (term12 - twoa * ym * py)
                self._chi1[iwfn, :nprims, 2] = xyexp * (term13 - twoa * zn * pz)

                # Second derivatives (if ider >= 2)
                if ider >= 2:
                    twoa_chi = twoa * self._chi[iwfn, :nprims]  # Shape: (nprims,)

                    # xx, yy, zz
                    self._chi2[iwfn, :nprims, 0] = gpow(px, l - 2) * yzexp * l * (l - 1) - twoa_chi * (
                        2.0 * l + 1.0 - twoa * px**2
                    )
                    self._chi2[iwfn, :nprims, 3] = gpow(py, m - 2) * xzexp * m * (m - 1) - twoa_chi * (
                        2.0 * m + 1.0 - twoa * py**2
                    )
                    self._chi2[iwfn, :nprims, 5] = gpow(pz, n - 2) * xyexp * n * (n - 1) - twoa_chi * (
                        2.0 * n + 1.0 - twoa * pz**2
                    )

                    expee = twoa * expon  # Shape: (nprims,)
                    foura_two_chi = 4.0 * alpha**2 * self._chi[iwfn, :nprims]  # Shape: (nprims,)

                    # xy
                    self._chi2[iwfn, :nprims, 1] = (
                        term11 * term12 * zn * expon
                        - term12 * xl * px * zn * expee
                        - term11 * ym * py * zn * expee
                        + px * py * foura_two_chi
                    )

                    # xz
                    self._chi2[iwfn, :nprims, 2] = (
                        term11 * term13 * ym * expon
                        - term13 * xl * px * ym * expee
                        - term11 * zn * pz * ym * expee
                        + px * pz * foura_two_chi
                    )

                    # yz
                    self._chi2[iwfn, :nprims, 4] = (
                        term12 * term13 * xl * expon
                        - term13 * ym * py * xl * expee
                        - term12 * zn * pz * xl * expee
                        + py * pz * foura_two_chi
                    )

        return self._chi, self._chi1, self._chi2

    def _gen_denmat(self):
        """Generate the density matrix for the given point.

        Denmat effectively computes:
            D_pq = sum_i occ_i * C_ip * C_iq
        """

        for iwfn in range(self._wfns_rep.nwfns):
            self._denmat[iwfn, :, :] = np.einsum("i,ip,iq->pq", self._occ[iwfn], self._mocs[iwfn], self._mocs[iwfn])

    def read_vib_file(self, input_file: str):
        """
        Read the log file from the optimization procedure.
        Expected to include sufficient information to generate the MSDA.
        """
        raise NotImplementedError

    def read_msda_matrix(self, msda_file: str):
        """Read the MSDA matrix from a file."""

        raise NotImplementedError

    def rho(self, x: float, y: float, z: float) -> float:
        """Generate the ED at a point.

        Args:
            x, y, z: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

        self._gen_chi(x, y, z, ider=0)

        rhov = 0.0
        for iwfn in range(self._wfns_rep.nwfns):
            rhov += float(
                np.sum(self._denmat[iwfn, ...] * self._chi[iwfn, :, np.newaxis] * self._chi[iwfn, np.newaxis, :])
            )

        rhov /= self._wfns_rep.nwfns

        return rhov

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args:
            x, y, z: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_chi(x, y, z, ider=1)

        gradv = np.zeros(3, dtype=float)
        for iwfn in range(self._wfns_rep.nwfns):

            # Compute pairwise products of _chi and _chi1
            chi_i_chi1_j = np.einsum("i,jk->ijk", self._chi[iwfn, ...], self._chi1[iwfn, ...])
            chi_j_chi1_i = np.einsum("j,ik->ijk", self._chi[iwfn, ...], self._chi1[iwfn, ...])

            # Combine the contributions to the gradient
            for dim in range(3):  # Iterate over x, y, z dimensions
                gradv[dim] += np.sum(self._denmat[iwfn] * (chi_i_chi1_j[:, :, dim] + chi_j_chi1_i[:, :, dim]))

        gradv /= self._wfns_rep.nwfns

        return gradv

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args:
            x, y, z: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_chi(x, y, z, ider=2)

        dxx, dyy, dzz, dxy, dxz, dyz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        hessv = np.zeros(6, dtype=float)

        for iwfn in range(self._wfns_rep.nwfns):

            # Diagonal terms
            dxx = np.sum(
                self._denmat
                * (
                    self._chi[iwfn, :, np.newaxis] * self._chi2[iwfn, np.newaxis, :, 0]
                    + 2.0 * self._chi1[iwfn, :, 0][:, np.newaxis] * self._chi1[iwfn, :, 0][np.newaxis, :]
                    + self._chi2[iwfn, :, 0][:, np.newaxis] * self._chi[iwfn, np.newaxis, :]
                )
            )
            dyy = np.sum(
                self._denmat
                * (
                    self._chi[iwfn, :, np.newaxis] * self._chi2[iwfn, np.newaxis, :, 3]
                    + 2.0 * self._chi1[iwfn, :, 1][:, np.newaxis] * self._chi1[iwfn, :, 1][np.newaxis, :]
                    + self._chi2[iwfn, :, 3][:, np.newaxis] * self._chi[iwfn, np.newaxis, :]
                )
            )
            dzz = np.sum(
                self._denmat
                * (
                    self._chi[iwfn, :, np.newaxis] * self._chi2[iwfn, np.newaxis, :, 5]
                    + 2.0 * self._chi1[iwfn, :, 2][:, np.newaxis] * self._chi1[iwfn, :, 2][np.newaxis, :]
                    + self._chi2[iwfn, :, 5][:, np.newaxis] * self._chi[iwfn, np.newaxis, :]
                )
            )

            # Off-diagonal terms
            dxy = np.sum(
                self._denmat
                * (
                    self._chi[iwfn, :, np.newaxis] * self._chi2[iwfn, np.newaxis, :, 1]
                    + self._chi1[iwfn, :, 0][:, np.newaxis] * self._chi1[iwfn, :, 1][np.newaxis, :]
                    + self._chi1[iwfn, :, 1][:, np.newaxis] * self._chi1[iwfn, :, 0][np.newaxis, :]
                    + self._chi2[iwfn, :, 1][:, np.newaxis] * self._chi[iwfn, np.newaxis, :]
                )
            )
            dxz = np.sum(
                self._denmat
                * (
                    self._chi[iwfn, :, np.newaxis] * self._chi2[iwfn, np.newaxis, :, 2]
                    + self._chi1[iwfn, :, 0][:, np.newaxis] * self._chi1[iwfn, :, 2][np.newaxis, :]
                    + self._chi1[iwfn, :, 2][:, np.newaxis] * self._chi1[iwfn, :, 0][np.newaxis, :]
                    + self._chi2[iwfn, :, 2][:, np.newaxis] * self._chi[iwfn, np.newaxis, :]
                )
            )
            dyz = np.sum(
                self._denmat
                * (
                    self._chi[iwfn, :, np.newaxis] * self._chi2[iwfn, np.newaxis, :, 4]
                    + self._chi1[iwfn, :, 1][:, np.newaxis] * self._chi1[iwfn, :, 2][np.newaxis, :]
                    + self._chi1[iwfn, :, 2][:, np.newaxis] * self._chi1[iwfn, :, 1][np.newaxis, :]
                    + self._chi2[iwfn, :, 4][:, np.newaxis] * self._chi[iwfn, np.newaxis, :]
                )
            )

            # Combine into Hessian vector
            hessv = np.array([dxx, dyy, dzz, dxy, dxz, dyz], dtype=float)

        hessv /= self._wfns_rep.nwfns

        return hessv


def _tst():  # pragma: no cover
    # pylint: disable=all
    # This is a test area for validating work above.
    import argparse
    import sys

    from .edwfn import EDWfn

    # Get one or more input files.
    parser = argparse.ArgumentParser(description="Test wfn reading.")
    parser.add_argument("-i", "--input", type=str, help="Input wfn file(s)")
    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        sys.exit(1)

    with open(args.input, "r") as finp:
        wfn_file = finp.readlines()[0].strip()

    edwfn = EDWfn(wfn_file)
    edwfns = EDWfns(args.input)

    print(edwfns.atpos)
    print(edwfns.atnames)

    rho_gs = edwfn.rho(0.0, 0.0, 0.0)
    rho_at = edwfns.rho(0.0, 0.0, 0.0)
    print(f"|rho_at| ..... {rho_at:16.12f}")
    print(f"|rho_gs| ..... {rho_gs:16.12f}")

    grad_gs = edwfn.grad(0.0, 0.0, 0.0)
    grad_at = edwfns.grad(0.0, 0.0, 0.0)
    print(f"|grad_at| .... {' '.join(f'{_g:16.12f}' for _g in grad_at)}")
    print(f"|grad_gs| .... {' '.join(f'{_g:16.12f}' for _g in grad_gs)}")

    hess_gs = edwfn.hess(0.0, 0.0, 0.0)
    hess_at = edwfns.hess(0.0, 0.0, 0.0)
    print(f"|hess_at| .... {' '.join(f'{_g:16.12f}' for _g in hess_at)}")
    print(f"|hess_gs| .... {' '.join(f'{_g:16.12f}' for _g in hess_gs)}")


if __name__ == "__main__":
    _tst()
