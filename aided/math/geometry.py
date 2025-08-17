"""
Non-trivial geometric functions and operations.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from aided import np, npt

from typing import Tuple


def distance_from_point_to_line(x: npt.NDArray, a: npt.NDArray, b: npt.NDArray) -> float:
    """Calculate the distance from a point to a line.

    Args:
        x: The point in question.
        a, b: Two points forming a line.

    Returns:
        d: The distance from the point to the line
    """

    # Vector from a to b
    ab = (b - a).flatten()

    # Vector from a to x
    ax = (x - a).flatten()

    # Cross product of ab and ax
    cross_product = np.cross(ab, ax)

    # Magnitude of the cross product
    cross_product_magnitude = np.linalg.norm(cross_product)

    # Magnitude of the vector ab
    ab_magnitude = np.linalg.norm(ab)

    # Shortest distance from point x to the line
    d = cross_product_magnitude / ab_magnitude

    return float(d)


def generate_spherical_grid(
    position: npt.NDArray, radius: float, ntheta: int, nphi: int
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Generates a spherical grid from a position, radius, and number of spherical angles.

    Args:
        position: The center of the sphere.
        radius: The radius of the sphere.
        ntheta: The number of polar angles (0 <= theta <= pi).
        nphi: The number of azimuthal angles (0 <= phi < 2pi).

    Returns:
        pts: The Cartesian coordinates of the points on the sphere.
        thetas: The polar angles of the points on the sphere.
        phis: The azimuthal angles of the points on the sphere.
    """
    # Flatten position to avoid shape issues
    pos = position.flatten()
    
    pts_list = []
    thetas_list = []
    phis_list = []
    for theta in np.linspace(0, np.pi, ntheta):
        for phi in np.linspace(0, 2 * np.pi, nphi):
            x = pos[0] + radius * np.sin(theta) * np.cos(phi)
            y = pos[1] + radius * np.sin(theta) * np.sin(phi)
            z = pos[2] + radius * np.cos(theta)
            if (x, y, z) in pts_list:
                continue
            pts_list.append((x, y, z))
            thetas_list.append(theta)
            phis_list.append(phi)
    pts = np.array(pts_list)
    thetas = np.array(thetas_list)
    phis = np.array(phis_list)

    return pts, thetas, phis
