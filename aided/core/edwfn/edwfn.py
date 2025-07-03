"""
aided.core.edwfn.edwfn

General Electron Density class for both dynamic and static.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from numpy.typing import NDArray

from aided import np
from aided.core.edrep import EDRep
from aided.io.read_wfn import read_wfn_file
from aided.core._edwfn import gen_gs


class EDWfn(EDRep):
    """
    Electron Density Representation from a single .wfn file.
    """

    def __init__(self, wfn_file: str):
        super().__init__(input_file=wfn_file)

        self._denmat: NDArray[np.float64]
        self._gs: NDArray[np.float64]
        self._gs1: NDArray[np.float64]
        self._gs2: NDArray[np.float64]
        self._occ: NDArray[np.float64]

        # Read the wfn file.
        self._wfn_rep = read_wfn_file(wfn_file)

        self._gs = np.zeros(self._wfn_rep.nprims, dtype=np.float64)
        self._gs1 = np.zeros((self._wfn_rep.nprims, 3), dtype=np.float64)
        self._gs2 = np.zeros((self._wfn_rep.nprims, 6), dtype=np.float64)
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

    def _gen_gs(self, x: float, y: float, z: float, ider: int) -> bool:
        """Generate the gs matrix for the given point.

        Skip this if the point is the same as the last point.

        Args:
            x, y, z: Cartesian points in global space.
            ider: Derivative order.

        Return: True if the gs matrix was generated, False otherwise.
        """

        did_compute, self._last_point, self._last_der = gen_gs(
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
            self._gs,
            self._gs1,
            self._gs2,
        )
        return did_compute

    def _gen_denmat(self):
        """Generate the density matrix for the given point.

        Denmat effectively computes:
            D_pq = sum_i occ_i * C_ip * C_iq
        """

        self._denmat = np.einsum("i,ip,iq->pq", self._occ, self._mocs, self._mocs)

    def rho(self, x: float, y: float, z: float) -> float:
        """Calculate the electron density at a given point."""
        raise NotImplementedError("Rho is not implemented for EDWfn.")

    def grad(self, x: float, y: float, z: float) -> NDArray[np.float64]:
        """Calculate the gradient of the electron density at a given point."""
        raise NotImplementedError("Grad is not implemented for EDWfn.")

    def hess(self, x: float, y: float, z: float) -> NDArray[np.float64]:
        """Calculate the Hessian of the electron density at a given point."""
        raise NotImplementedError("Hess is not implemented for EDWfn.")
