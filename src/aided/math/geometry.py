"""
Non-trivial geometric functions and operations.
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
