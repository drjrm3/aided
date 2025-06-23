"""
aided.core.edwfn.edwfn_dynamic

Dynamic Electron Density manifestation.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from aided import np
from .edwfn import EDWfn
from aided.io.vib.reader import gen_msda


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
        msda: str | None = None,
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


    def rho(self, x: float, y: float, z: float) -> float:
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

        self._gen_gs(x, y, z, ider=0)

        rhov = 0

        return rhov

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_gs(x, y, z, ider=1)

        gradv = np.zeros(3, dtype=np.float64)

        return gradv

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_gs(x, y, z, ider=2)

        hessv = np.zeros(6, dtype=np.float64)

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
