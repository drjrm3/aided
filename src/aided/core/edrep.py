"""
aided.core.EDRep

Electron Density Representation abstract class.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import List, Tuple

from scipy.optimize import minimize

from .. import np
from .units import Units
from ..math.geometry import distance_from_point_to_line


class EDRepType(Enum):
    WFN = 0
    RDF = 1
    NRDF = 2


class EDRep(metaclass=ABCMeta):
    """
    Electron Density Representation.

    High level class which represents the information needed to express the ED but is abstracted
    away from the type of file.

    Types of representations may include:
        - .wfn files (AIMFile)
    """

    def __init__(self, input_file: str):
        self._edrep_type = None
        self._units = Units.BOHR

    @property
    @abstractmethod
    def atpos(self) -> np.ndarray:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def atnames(self) -> np.ndarray:  # pragma: no cover
        pass

    @property
    def units(self):
        return self._units

    @property
    def in_au(self):
        return self._units == Units.BOHR

    @abstractmethod
    def rho(self, x: float, y: float, z: float) -> float:  # pragma: no cover
        """Generate the ED at a point.

        Args: Cartesian points in global space.

        Returns: Value of ED in chosen units.
        """

    @abstractmethod
    def grad(self, x: float, y: float, z: float) -> np.ndarray:  # pragma: no cover
        """Generate the Gradient of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 3 elements: dx, dy, dz
        """

    @abstractmethod
    def hess(self, x: float, y: float, z: float) -> np.ndarray:  # pragma: no cover
        """Generate the Hessian of the ED at a point.

        Args: Cartesian points in global space.

        Returns: Array of 6 elements: dxdx, dydy, dzdz, dxdy, dxdz, dydz.
        """

    def read_vib_file(self, input_file: str):
        """
        Read the log file from the optimization procedure.
        Expected to include sufficient information to generate the MSDA.
        """
        raise NotImplementedError

    def read_msda_matrix(self, msda_file: str):
        """Read the MSDA matrix from a file."""

        raise NotImplementedError

    def bader_surface_of_atom(
        self,
        atom_name: str,
        radius: float = 0.5,
        ntheta: int = 10,
        max_distance: float = 2.5,
        step_size: float = 0.1,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a Bader Surface for an atom.

        Args:
            atom_name: Name of the atom to generate the surface for.
            radius: Radius to generate the surface in Bohr.
            ntheta: Number of points to generate in the theta direction.
            max_distance: The max distance (in Bohr) from the atom to find a point on the surface.
            step_size: The step size to take when searching for the surface.
            tol: The tolerance to use when searching for the surface.

        Returns:
            thetas: Array of theta values.
            phis: Array of phi values.
            bader_surface: Numpy array of points on the surface.
        """

        # Find the atom position (atpos) of the atom.
        atom_position = np.zeros(3)
        for iat, atpos in enumerate(self.atpos):
            if self.atnames[iat] == atom_name:
                atom_position = atpos
                break

        # Generating starting points on a sphere where theta is in [0, pi), phi is in [0, 2pi),
        # the number of points for theta is ntheta and the number of points for phi is 2 * ntheta.
        nphi = 2 * ntheta
        thetas = np.linspace(0, np.pi, ntheta)
        phis = np.linspace(0, 2 * np.pi, nphi)
        surface = np.zeros((ntheta, nphi, 3))

        for itheta, theta in enumerate(thetas):
            for iphi, phi in enumerate(phis):
                x = atom_position[0] + radius * np.sin(theta) * np.cos(phi)
                y = atom_position[1] + radius * np.sin(theta) * np.sin(phi)
                z = atom_position[2] + radius * np.cos(theta)
                surface[itheta, iphi] = self._find_bader_surface_point(
                    np.array([x, y, z]), radius, theta, phi, max_distance, step_size, tol
                )

        return thetas, phis, surface

    def _find_bader_surface_point(
        self,
        start_pos: np.ndarray,
        radius: float,
        theta: float,
        phi: float,
        max_distance: float,
        step_size: float,
        tol: float,
    ) -> np.ndarray:
        """Find a point on the Bader Surface.

        Args:
            start_pos: Starting position to search from.
            radius: The initial radial distance away from the atom.
            theta: The theta value of the starting position.
            phi: The phi value of the starting position.
            max_distance: The max distance (in Bohr) from the atom to find a point on the surface.
            step_size: The step size to take when searching for the surface.
            tol: The tolerance to use when searching for the surface.

        Returns:
            point: The point on the Bader Surface.
        """

        point = start_pos

        # Get this atom name and position by finding the closest atom to the start position.
        this_atom_name = ""
        this_atom_position = np.zeros(3)
        for iat, atpos in enumerate(self.atpos):
            distance = np.linalg.norm(atpos - start_pos)
            if distance < tol:
                this_atom_name = self.atnames[iat]
                this_atom_position = atpos
                break

        # Continue to step away from the atom until the gradien trace finds a different atom.
        last_point = point
        _atom_name = this_atom_name
        while _atom_name == this_atom_name:
            last_point = point
            _atom_name, _ = self.trace_gradient_to_atom(point[0], point[1], point[2])
            # If we did not find 'this atom' then we have crossed the Bader surface. In this case
            # we increment the point a distnace of 'step_size' in the direction of (theta, phi).
            if _atom_name == this_atom_name:
                radius += step_size
                point = np.array(
                    [
                        this_atom_position[0] + radius * np.sin(theta) * np.cos(phi),
                        this_atom_position[1] + radius * np.sin(theta) * np.sin(phi),
                        this_atom_position[2] + radius * np.cos(theta),
                    ]
                )

        # Now we must search between point and last_point to find the Bader surface unti othe distance
        # between the two points is less than 'tol'.
        while np.linalg.norm(point - last_point) > tol:
            _atom_name, _ = self.trace_gradient_to_atom(point[0], point[1], point[2])
            if _atom_name == this_atom_name:
                last_point = point
                point = (point + last_point) / 2
            else:
                point = (point + last_point) / 2

        return point

    def trace_gradient_to_atom(
        self, x: float, y: float, z: float, method: str = "BFGS"
    ) -> Tuple[str, np.ndarray]:
        """Trace the gradient of the ED to an atom.

        Perform this by using scipy.optimize.minimize to find the location where the gradient is
        zero but the electron density (rho) is maximized.

        Args:
            x, y, z: Cartesian points in global space.

        Returns:
            atom_name: Name of the atom found.
            atom_position: Position of the atom found.
        """

        # intermediate_points = [np.array([x, y, z])]

        # Callback method to track points.
        # def callback(point):
        #    intermediate_points.append(point)

        def objective(point):
            return -self.rho(point[0], point[1], point[2])

        def grad(point):
            return -self.grad(point[0], point[1], point[2])

        found_position = minimize(
            fun=objective,
            x0=np.array([x, y, z]),
            jac=grad,
            method=method,
        )

        # Find the atom position (atpos) which is closest to the atom found.
        atom_name = ""
        atom_position = np.zeros(3)
        atom_distance = 1e6
        for iat, atpos in enumerate(self.atpos):
            distance = np.linalg.norm(atpos - found_position.x)
            if distance < atom_distance:
                atom_distance = distance
                atom_name = self.atnames[iat]
                atom_position = atpos

        return atom_name, atom_position

    def bcp(self, x: float, y: float, z: float, method: str = "BFGS"):
        """Find the BCP at a point.

        Given a starting value of x, y, z, find the Bond Critical Point which has properties of:
            - Being a local minima in the gradient field.

        Args:
            x, y, z: Cartesian points in global space.
            method: Optimization method to use (BFGS, CG, Netwon-CG, etc.)

        Returns: Array of 3 elements: x, y, z
        """

        intermediate_points = [np.array([x, y, z])]

        # Callback method to track points.
        def callback(point):
            intermediate_points.append(point)

        # Objective function to minimize.
        def objective(point):
            return np.linalg.norm(self.grad(point[0], point[1], point[2]))

        bcp = minimize(
            fun=objective,
            x0=np.array([x, y, z]),
            method=method,
            callback=callback,
        )

        return bcp, intermediate_points

    def find_bcps(self, max_pt_distance: float = 0.1) -> Tuple[List, List]:
        """Find all BCPs in the system.

        Args:
            max_pt_distance: Max distance to allow from a BPC to the line connecting two atoms.

        Returns:
            bcps: List of BCPs found.
            bond_pairs: List of bond pairs found.
        """

        bcps = []
        bond_pairs = []
        intermediate_points = []

        for iat, (iat_name, iat_pos) in enumerate(zip(self.atnames, self.atpos)):
            for jat_name, jat_pos in zip(self.atnames[iat:], self.atpos[iat:]):
                if iat_name == jat_name:
                    continue

                # TODO: Check distance between nuclei and see if this is reasonable for a bond.

                # Get starting position for BCP.
                start_pos = (iat_pos + jat_pos) / 2

                # Find the BCP from the starting position.
                bcp, _pts = self.bcp(start_pos[0], start_pos[1], start_pos[2])

                # TODO: Check distance between BCP and nuclei and see if this is reasonable.

                # Get the distance from bcp.x to the line connecting the two atoms.
                bcp_dist = distance_from_point_to_line(bcp.x, iat_pos, jat_pos)
                if bcp_dist > max_pt_distance:
                    continue

                at1_name, at2_name = min(iat_name, jat_name), max(iat_name, jat_name)
                if (at1_name, at2_name) in bond_pairs:
                    continue

                bond_pairs.append((at1_name, at2_name))
                bcps.append(bcp)
                intermediate_points.append(_pts)

        return bcps, bond_pairs


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
    start_pos = edwfn.atpos[0]
    bcp, _pts = edwfn.bcp(0, 0, 0)
    print(f"BCP: {bcp}")


if __name__ == "__main__":
    _tst()
