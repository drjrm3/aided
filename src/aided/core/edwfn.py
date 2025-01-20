"""
aided.core.EDWfn

Electron Density Manfestation abstract class.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from .edrep import EDRep

from .. import LMNS, np
from ..fio.read_wfn import read_wfn_file
from ..math.primitives import gpow


class EDWfn(EDRep):
    """
    Electron Density Representation from a single .wfn file.
    """

    def __init__(self, wfn_file: str):
        super().__init__(input_file=wfn_file)

        # Read the wfn file.
        self._wfn_rep = read_wfn_file(wfn_file)

        self._chi = np.zeros(self._wfn_rep.nprims, dtype=float)
        self._chi1 = np.zeros((self._wfn_rep.nprims, 3), dtype=float)
        self._chi2 = np.zeros((self._wfn_rep.nprims, 6), dtype=float)
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

        for iprim in range(self._wfn_rep.nprims):

            # Spherical Harmonics.
            l, m, n = LMNS[self._wfn_rep.types[iprim]]
            center = self._wfn_rep.centers[iprim]
            cx, cy, cz = self._wfn_rep.atpos[center][:]

            # Coordinates relative to the atomic center to evaluate property for this primitive.
            px, py, pz = x - cx, y - cy, z - cz

            # Argument for this Gaussian primitive.
            alpha = self._wfn_rep.expons[iprim]
            expon = np.exp(-alpha * (px**2 + py**2 + pz**2))
            xl, ym, zn = np.power([px, py, pz], [l, m, n])

            self._chi[iprim] = xl * ym * zn * expon

            # First derivative.
            if ider >= 1:
                twoa = 2.0 * alpha

                term11 = l * gpow(px, l - 1)
                term12 = m * gpow(py, m - 1)
                term13 = n * gpow(pz, n - 1)

                xyexp = xl * ym * expon
                xzexp = xl * zn * expon
                yzexp = ym * zn * expon

                self._chi1[iprim, 0] = yzexp * (term11 - twoa * xl * px)
                self._chi1[iprim, 1] = xzexp * (term12 - twoa * ym * py)
                self._chi1[iprim, 2] = xyexp * (term13 - twoa * zn * pz)

                # Second derivative.
                if ider >= 2:
                    twoa_chi = twoa * self._chi[iprim]

                    # xx
                    self._chi2[iprim, 0] = gpow(px, l - 2) * yzexp * l * (l - 1) - twoa_chi * (
                        2.0 * l + 1.0 - twoa * px * px
                    )

                    # yy
                    self._chi2[iprim, 3] = gpow(py, m - 2) * xzexp * m * (m - 1) - twoa_chi * (
                        2.0 * m + 1.0 - twoa * py * py
                    )

                    # zz
                    self._chi2[iprim, 5] = gpow(pz, n - 2) * xyexp * n * (n - 1) - twoa_chi * (
                        2.0 * n + 1.0 - twoa * pz * pz
                    )

                    expee = twoa * expon
                    foura_two_chi = 4.0 * alpha * alpha * self._chi[iprim]

                    # xy
                    self._chi2[iprim, 1] = (
                        term11 * term12 * zn * expon
                        + -term12 * xl * px * zn * expee
                        + -term11 * ym * py * zn * expee
                        + px * py * foura_two_chi
                    )

                    # xz
                    self._chi2[iprim, 2] = (
                        term11 * term13 * ym * expon
                        + -term13 * xl * px * ym * expee
                        + -term11 * zn * pz * ym * expee
                        + px * pz * foura_two_chi
                    )

                    # yz
                    self._chi2[iprim, 4] = (
                        term12 * term13 * xl * expon
                        + -term13 * ym * py * xl * expee
                        + -term12 * zn * pz * xl * expee
                        + py * pz * foura_two_chi
                    )

    def _gen_denmat(self):
        """Generate the density matrix for the given point."""

        # TODO: Vectorize this.
        for iprim in range(self._wfn_rep.nprims):
            for jprim in range(self._wfn_rep.nprims):
                self._denmat[iprim, jprim] = 0.0
                for imo in range(self._wfn_rep.nmos):
                    self._denmat[iprim, jprim] += (
                        self._wfn_rep.occs[imo] * self._wfn_rep.coeffs[imo, iprim] * self._wfn_rep.coeffs[imo, jprim]
                    )

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

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

        self._gen_chi(x, y, z, ider=0)

        rhov = 0.0
        for iprim in range(self._wfn_rep.nprims):
            for jprim in range(self._wfn_rep.nprims):
                rhov += self._denmat[iprim, jprim] * self._chi[iprim] * self._chi[jprim]

        return rhov

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_chi(x, y, z, ider=1)

        gradv = np.zeros(3, dtype=float)

        for iprim in range(self._wfn_rep.nprims):
            chi_i = self._chi[iprim]

            chi_xi, chi_yi, chi_zi = self._chi1[iprim, :]

            for jprim in range(self._wfn_rep.nprims):
                chi_j = self._chi[jprim]

                chi_xj, chi_yj, chi_zj = self._chi1[jprim, :]

                gradv[0] += self._denmat[iprim, jprim] * (chi_i * chi_xj + chi_j * chi_xi)
                gradv[1] += self._denmat[iprim, jprim] * (chi_i * chi_yj + chi_j * chi_yi)
                gradv[2] += self._denmat[iprim, jprim] * (chi_i * chi_zj + chi_j * chi_zi)

        return gradv

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_chi(x, y, z, ider=2)

        dxx, dyy, dzz, dxy, dxz, dyz = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        for iprim in range(self._wfn_rep.nprims):
            chi_i = self._chi[iprim]

            chi_xi = self._chi1[iprim, 0]
            chi_yi = self._chi1[iprim, 1]
            chi_zi = self._chi1[iprim, 2]

            chi_xxi = self._chi2[iprim, 0]
            chi_xyi = self._chi2[iprim, 1]
            chi_xzi = self._chi2[iprim, 2]
            chi_yyi = self._chi2[iprim, 3]
            chi_yzi = self._chi2[iprim, 4]
            chi_zzi = self._chi2[iprim, 5]

            for jprim in range(self._wfn_rep.nprims):
                chi_j = self._chi[jprim]

                chi_xj = self._chi1[jprim, 0]
                chi_yj = self._chi1[jprim, 1]
                chi_zj = self._chi1[jprim, 2]

                chi_xxj = self._chi2[jprim, 0]
                chi_xyj = self._chi2[jprim, 1]
                chi_xzj = self._chi2[jprim, 2]
                chi_yyj = self._chi2[jprim, 3]
                chi_yzj = self._chi2[jprim, 4]
                chi_zzj = self._chi2[jprim, 5]

                # fmt: off
                dxx += self._denmat[iprim, jprim] * (
                    chi_i * chi_xxj + 2.0 * chi_xi * chi_xj + chi_xxi * chi_j
                )
                dyy += self._denmat[iprim, jprim] * (
                    chi_i * chi_yyj + 2.0 * chi_yi * chi_yj + chi_yyi * chi_j
                )
                dzz += self._denmat[iprim, jprim] * (
                    chi_i * chi_zzj + 2.0 * chi_zi * chi_zj + chi_zzi * chi_j
                )
                dxy += self._denmat[iprim, jprim] * (
                    chi_i * chi_xyj + chi_yi * chi_xj + chi_xi * chi_yj + chi_xyi * chi_j
                )
                dxz += self._denmat[iprim, jprim] * (
                    chi_i * chi_xzj + chi_xi * chi_zj + chi_zi * chi_xj + chi_xzi * chi_j
                )
                dyz += self._denmat[iprim, jprim] * (
                    chi_i * chi_yzj + chi_yi * chi_zj + chi_zi * chi_yj + chi_yzi * chi_j
                )
                # fmt: on

        hessv = np.array([dxx, dyy, dzz, dxy, dxz, dyz], dtype=float)

        return hessv


if __name__ == "__main__":
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

    if len(args.input) == 1:
        edwfn = EDWfn(args.input[0])
        print("")
        rho = edwfn.rho(0.0, 0.0, 0.0)
        print(f"{edwfn.rho(0.0, 0.0, 0.0)=}")
        grad = edwfn.grad(0.0, 0.0, 0.0)
        print(f"{edwfn.grad(0.0, 0.0, 0.0)=}")
        # hess = edwfn.hess(0.0, 0.0, 0.0)
        hess = edwfn.hess(0.0, 0.0, 0.0)
        print(f"{edwfn.hess(0.0, 0.0, 0.0)=}")
        print(f"{np.sum(edwfn.hess(0.0, 0.0, 0.0)[:3])=}")
