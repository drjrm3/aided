"""
aided.core.edwfn.static

Static Electron Density manifestation.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from numba import njit
from numpy.typing import NDArray

from aided import np
from .edwfn import EDWfn


@njit(fastmath=True, cache=True)
def numba_rho(denmat: NDArray, gs: NDArray) -> float:  # pragma: no cover
    """Compute the density using numba

    Args:
        denmat: Density matrix.
        gs: GTO basis functions.

    Returns:
        Density value.
    """
    rho = 0.0
    n = gs.shape[0]
    for i in range(n):
        for j in range(n):
            rho += denmat[i, j] * gs[i] * gs[j]
    return rho


@njit(fastmath=True, cache=True)
def numba_grad(denmat: NDArray, gs: NDArray, gs1: NDArray) -> NDArray:  # pragma: no cover
    """Compute the gradient using numba

    Args:
        denmat: Density matrix.
        gs: GTO basis functions.
        gs1: First derivative of gs.

    Returns:
        Gradient vector.
    """
    grad = np.zeros(3)
    n = gs.shape[0]
    for i in range(n):
        for j in range(n):
            for dim in range(3):
                grad[dim] += denmat[i, j] * (gs[i] * gs1[j, dim] + gs[j] * gs1[i, dim])
    return grad


@njit(fastmath=True, cache=True)
def numba_hess(
    denmat: NDArray, gs: NDArray, gs1: NDArray, gs2: NDArray
) -> NDArray:  # pragma: no cover
    """Compute the hessian using numba

    Args:
        denmat: Density matrix.
        gs: GTO basis functions.
        gs1: First derivative of gs.
        gs2: Second derivative of gs.

    Returns:
        Hessian matrix.
    """
    hess = np.zeros(6)
    n = gs.shape[0]

    for i in range(n):
        for j in range(n):
            hess[0] += denmat[i, j] * (
                gs[i] * gs2[j, 0] + 2 * gs1[i, 0] * gs1[j, 0] + gs2[i, 0] * gs[j]
            )
            hess[1] += denmat[i, j] * (
                gs[i] * gs2[j, 3] + 2 * gs1[i, 1] * gs1[j, 1] + gs2[i, 3] * gs[j]
            )
            hess[2] += denmat[i, j] * (
                gs[i] * gs2[j, 5] + 2 * gs1[i, 2] * gs1[j, 2] + gs2[i, 5] * gs[j]
            )
            hess[3] += denmat[i, j] * (
                gs[i] * gs2[j, 1]
                + gs1[i, 0] * gs1[j, 1]
                + gs1[i, 1] * gs1[j, 0]
                + gs2[i, 1] * gs[j]
            )
            hess[4] += denmat[i, j] * (
                gs[i] * gs2[j, 2]
                + gs1[i, 0] * gs1[j, 2]
                + gs1[i, 2] * gs1[j, 0]
                + gs2[i, 2] * gs[j]
            )
            hess[5] += denmat[i, j] * (
                gs[i] * gs2[j, 4]
                + gs1[i, 1] * gs1[j, 2]
                + gs1[i, 2] * gs1[j, 1]
                + gs2[i, 4] * gs[j]
            )

    return hess


class EDWfnStatic(EDWfn):
    """
    Electron Density Static manifestation from a single .wfn file.
    """

    def __init__(self, wfn_file: str):
        super().__init__(wfn_file)

    def rho(self, x: float, y: float, z: float) -> float:
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

        self._gen_gs(x, y, z, ider=0)

        rhov = numba_rho(self._denmat, self._gs)

        return rhov

    def grad(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

        self._gen_gs(x, y, z, ider=1)

        gradv = numba_grad(self._denmat, self._gs, self._gs1)

        return gradv

    def hess(self, x: float, y: float, z: float) -> np.ndarray:
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

        self._gen_gs(x, y, z, ider=2)

        hessv = numba_hess(self._denmat, self._gs, self._gs1, self._gs2)
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
