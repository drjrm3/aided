"""
Non-trivial geometric functions and operations.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from .. import np, npt


def distance_from_point_to_line(x: npt.NDArray, a: npt.NDArray, b: npt.NDArray) -> float:
    """Calculate the distance from a point to a line.

    Args:
        x: The point in question.
        a, b: Two points forming a line.

    Returns:
        d: The distance from the point to the line
    """

    # Vector from a to b
    ab = b - a

    # Vector from a to x
    ax = x - a

    # Cross product of ab and ax
    cross_product = np.cross(ab, ax)

    # Magnitude of the cross product
    cross_product_magnitude = np.linalg.norm(cross_product)

    # Magnitude of the vector ab
    ab_magnitude = np.linalg.norm(ab)

    # Shortest distance from point x to the line
    d = cross_product_magnitude / ab_magnitude

    return float(d)


def spherical_angles_from_vector(v: npt.NDArray) -> tuple[float, float]:
    """Convert a 3D vector to spherical angles.

    Args:
        v: A 3D vector of x, y, z coordinates.

    Returns:
        theta: The polar angle (angle from the z-axis).
        phi: The azimuthal angle (angle in the x-y plane from the x-axis).
    """
    x, y, z = v
    r = np.linalg.norm(v)
    # 0 <= theta <= pi
    theta = np.arccos(z / r)
    # 0 <= phi < 2pi
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi
    return theta, phi
