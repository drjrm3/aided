"""
aided.core.edwfns

Electron Density Representations from .wfn files.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from datetime import datetime
from typing import List
from numpy.typing import NDArray

from .edrep import EDRep

from .. import LMNS, np, npt
from ..io.read_wfn import read_wfn_files
from ..math.primitives import gpow


def compute_hessian_batch(
    chi: npt.NDArray, chi1: npt.NDArray, chi2: npt.NDArray, denmat: npt.NDArray
) -> np.ndarray:
    """Compute the Hessian for a batch of wavefunctions.

    Args:
        chi: Chi matrix.
        chi1: First derivative of chi matrix.
        chi2: Second derivative of chi matrix.
        denmat: Density matrix.

    Returns:
        hessv: Hessian vector as diagonals [dxx, dyy, dzz, dxy, dxz, dyz].
    """

    # Compute diagonal terms
    hessv = np.zeros(6, dtype=float)
    hessv[0] = np.einsum(
        "ijk,ijk->",
        denmat,
        chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 0]
        + 2.0 * chi1[:, :, 0][:, :, np.newaxis] * chi1[:, :, 0][:, np.newaxis, :]
        + chi2[:, :, 0][:, :, np.newaxis] * chi[:, np.newaxis, :],
    )
    hessv[1] = np.einsum(
        "ijk,ijk->",
        denmat,
        chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 3]
        + 2.0 * chi1[:, :, 1][:, :, np.newaxis] * chi1[:, :, 1][:, np.newaxis, :]
        + chi2[:, :, 3][:, :, np.newaxis] * chi[:, np.newaxis, :],
    )
    hessv[2] = np.einsum(
        "ijk,ijk->",
        denmat,
        chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 5]
        + 2.0 * chi1[:, :, 2][:, :, np.newaxis] * chi1[:, :, 2][:, np.newaxis, :]
        + chi2[:, :, 5][:, :, np.newaxis] * chi[:, np.newaxis, :],
    )

    # Compute off-diagonal terms
    hessv[3] = np.einsum(
        "ijk,ijk->",
        denmat,
        chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 1]
        + chi1[:, :, 0][:, :, np.newaxis] * chi1[:, :, 1][:, np.newaxis, :]
        + chi1[:, :, 1][:, :, np.newaxis] * chi1[:, :, 0][:, np.newaxis, :]
        + chi2[:, :, 1][:, :, np.newaxis] * chi[:, np.newaxis, :],
    )
    hessv[4] = np.einsum(
        "ijk,ijk->",
        denmat,
        chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 2]
        + chi1[:, :, 0][:, :, np.newaxis] * chi1[:, :, 2][:, np.newaxis, :]
        + chi1[:, :, 2][:, :, np.newaxis] * chi1[:, :, 0][:, np.newaxis, :]
        + chi2[:, :, 2][:, :, np.newaxis] * chi[:, np.newaxis, :],
    )
    hessv[5] = np.einsum(
        "ijk,ijk->",
        denmat,
        chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 4]
        + chi1[:, :, 1][:, :, np.newaxis] * chi1[:, :, 2][:, np.newaxis, :]
        + chi1[:, :, 2][:, :, np.newaxis] * chi1[:, :, 1][:, np.newaxis, :]
        + chi2[:, :, 4][:, :, np.newaxis] * chi[:, np.newaxis, :],
    )

    return hessv


class EDWfns(EDRep):
    """
    Electron Density Representation from multiple .wfn file.
    """

    def __init__(self, wfn_file_list: str, io_procs: int = 1):
        """Initialization with list of .wfn files and number of processes to use.

        Args:
            wfn_file_list: List of .wfn files.
            io_procs: Number of proceess to use for io operations.
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
        self._wfns_rep = read_wfn_files(wfn_files, nprocs=io_procs)

        # Assumes that all atnames are the same.
        self._atnames = self._wfns_rep.atnames[0]

        # Assumes that all .wfns represent the same molecule and we want the averaged position.
        self._atpos = np.mean(self._wfns_rep.atpos, axis=0)

        # Initialize the chi matrices.
        self._chi = np.zeros((self._wfns_rep.nwfns, self._wfns_rep.nprims), dtype=float)
        self._chi1 = np.zeros((self._wfns_rep.nwfns, self._wfns_rep.nprims, 3), dtype=float)
        self._chi2 = np.zeros((self._wfns_rep.nwfns, self._wfns_rep.nprims, 6), dtype=float)
        self._denmat = np.zeros(
            (self._wfns_rep.nwfns, self._wfns_rep.nprims, self._wfns_rep.nprims), dtype=float
        )

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

    def __eq__(self, other) -> bool:
        """Equality comparison."""
        return self._wfns_rep == other._wfns_rep

    @property
    def atpos(self) -> np.ndarray:
        # Average the atomic positions.
        return self._atpos

    @property
    def atnames(self) -> np.ndarray:
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
                    self._chi2[iwfn, :nprims, 0] = gpow(px, l - 2) * yzexp * l * (
                        l - 1
                    ) - twoa_chi * (2.0 * l + 1.0 - twoa * px**2)
                    self._chi2[iwfn, :nprims, 3] = gpow(py, m - 2) * xzexp * m * (
                        m - 1
                    ) - twoa_chi * (2.0 * m + 1.0 - twoa * py**2)
                    self._chi2[iwfn, :nprims, 5] = gpow(pz, n - 2) * xyexp * n * (
                        n - 1
                    ) - twoa_chi * (2.0 * n + 1.0 - twoa * pz**2)

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
            self._denmat[iwfn, :, :] = np.einsum(
                "i,ip,iq->pq", self._occ[iwfn], self._mocs[iwfn], self._mocs[iwfn]
            )

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
                np.sum(
                    self._denmat[iwfn, ...]
                    * self._chi[iwfn, :, np.newaxis]
                    * self._chi[iwfn, np.newaxis, :]
                )
            )

        rhov /= self._wfns_rep.nwfns

        """
        # Pre-compute chi products once for all iwfn
        chi_product = self._chi[:, :, np.newaxis] * self._chi[:, np.newaxis, :]

        # Perform the matrix product for all iwfn at once
        rhov = np.einsum("ijk,ijk->i", self._denmat, chi_product)

        # Average across all iwfn
        rhov = np.sum(rhov) / self._wfns_rep.nwfns
        """

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
                gradv[dim] += np.sum(
                    self._denmat[iwfn] * (chi_i_chi1_j[:, :, dim] + chi_j_chi1_i[:, :, dim])
                )

        gradv /= self._wfns_rep.nwfns

        return gradv

    def hess(self, x: float, y: float, z: float, batch_size: int = 1000) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        This function can be very memory intensive, so delete partial computations.

        Args:
            x, y, z: Cartesian points in global space.
            batch_size: Number of wavefunctions to process at once. If zero, process all at once.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_chi(x, y, z, ider=2)

        hessv = np.zeros(6, dtype=float)

        # Extract the components for clarity
        chi = self._chi
        chi1 = self._chi1
        chi2 = self._chi2
        denmat = self._denmat
        nwfns = self._wfns_rep.nwfns

        if not batch_size:
            # This is fast but memory intensive for a large number of wavefunctions.
            # Compute diagonal terms
            hessv[0] = np.einsum(
                "ijk,ijk->",
                denmat,
                chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 0]
                + 2.0 * chi1[:, :, 0][:, :, np.newaxis] * chi1[:, :, 0][:, np.newaxis, :]
                + chi2[:, :, 0][:, :, np.newaxis] * chi[:, np.newaxis, :],
            )
            hessv[1] = np.einsum(
                "ijk,ijk->",
                denmat,
                chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 3]
                + 2.0 * chi1[:, :, 1][:, :, np.newaxis] * chi1[:, :, 1][:, np.newaxis, :]
                + chi2[:, :, 3][:, :, np.newaxis] * chi[:, np.newaxis, :],
            )
            hessv[2] = np.einsum(
                "ijk,ijk->",
                denmat,
                chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 5]
                + 2.0 * chi1[:, :, 2][:, :, np.newaxis] * chi1[:, :, 2][:, np.newaxis, :]
                + chi2[:, :, 5][:, :, np.newaxis] * chi[:, np.newaxis, :],
            )

            # Compute off-diagonal terms
            hessv[3] = np.einsum(
                "ijk,ijk->",
                denmat,
                chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 1]
                + chi1[:, :, 0][:, :, np.newaxis] * chi1[:, :, 1][:, np.newaxis, :]
                + chi1[:, :, 1][:, :, np.newaxis] * chi1[:, :, 0][:, np.newaxis, :]
                + chi2[:, :, 1][:, :, np.newaxis] * chi[:, np.newaxis, :],
            )
            hessv[4] = np.einsum(
                "ijk,ijk->",
                denmat,
                chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 2]
                + chi1[:, :, 0][:, :, np.newaxis] * chi1[:, :, 2][:, np.newaxis, :]
                + chi1[:, :, 2][:, :, np.newaxis] * chi1[:, :, 0][:, np.newaxis, :]
                + chi2[:, :, 2][:, :, np.newaxis] * chi[:, np.newaxis, :],
            )
            hessv[5] = np.einsum(
                "ijk,ijk->",
                denmat,
                chi[:, :, np.newaxis] * chi2[:, np.newaxis, :, 4]
                + chi1[:, :, 1][:, :, np.newaxis] * chi1[:, :, 2][:, np.newaxis, :]
                + chi1[:, :, 2][:, :, np.newaxis] * chi1[:, :, 1][:, np.newaxis, :]
                + chi2[:, :, 4][:, :, np.newaxis] * chi[:, np.newaxis, :],
            )

        else:
            # Process wavefunctions in batches.
            for start in range(0, nwfns, batch_size):
                end = min(start + batch_size, nwfns)

                # Slice the batch
                chi_batch = chi[start:end]
                chi1_batch = chi1[start:end]
                chi2_batch = chi2[start:end]
                denmat_batch = denmat[start:end]

                # Compute and accumulate the Hessian for this batch
                hessv += compute_hessian_batch(chi_batch, chi1_batch, chi2_batch, denmat_batch)

        # Normalize by the number of wavefunctions
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
    print(f"Reading wfn files ... ", end="")
    sys.stdout.flush()
    tic = datetime.now()
    edwfns = EDWfns(args.input, io_procs=4)
    toc = datetime.now()
    dif = (toc - tic).total_seconds()
    print(f"Done in {dif:16.12f} seconds")

    print(f"Static ... ")
    static_bcps, static_bonds = edwfn.find_bcps()
    for bond, bcp in zip(static_bonds, static_bcps):
        print(f"{bond} {bcp.x=}")

    print(f"Dynamic ... ")
    dynamic_bcps, dynamic_bonds = edwfns.find_bcps()
    for bond, bcp in zip(dynamic_bonds, dynamic_bcps):
        print(f"{bond} {bcp.x=}")

    """
    tic = datetime.now()
    rho_gs = edwfn.rho(0.0, 0.0, 0.0)
    rho_at = edwfns.rho(0.0, 0.0, 0.0)
    toc = datetime.now()
    dif = (toc - tic).total_seconds()

    print(f"|rho_at| ..... {rho_at:16.12f}")
    print(f"|rho_gs| ..... {rho_gs:16.12f}")
    print(f"Time taken ... {dif:16.12f} seconds")
    sys.stdout.flush()

    tic = datetime.now()
    grad_gs = edwfn.grad(0.0, 0.0, 0.0)
    grad_at = edwfns.grad(0.0, 0.0, 0.0)
    toc = datetime.now()
    dif = (toc - tic).total_seconds()
    print(f"|grad_at| .... {' '.join(f'{_g:16.12f}' for _g in grad_at)}")
    print(f"|grad_gs| .... {' '.join(f'{_g:16.12f}' for _g in grad_gs)}")
    print(f"Time taken ... {dif:16.12f} seconds")
    sys.stdout.flush()

    tic = datetime.now()
    hess_gs = edwfn.hess(0.0, 0.0, 0.0)
    hess_at = edwfns.hess(0.0, 0.0, 0.0)
    toc = datetime.now()
    dif = (toc - tic).total_seconds()
    print(f"|hess_at| .... {' '.join(f'{_g:16.12f}' for _g in hess_at)}")
    print(f"|hess_gs| .... {' '.join(f'{_g:16.12f}' for _g in hess_gs)}")
    print(f"Time taken ... {dif:16.12f} seconds")
    sys.stdout.flush()
    """


if __name__ == "__main__":  # pragma: no cover
    _tst()
