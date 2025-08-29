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
    pos = position.flatten()
    
    theta_values = np.linspace(0, np.pi, ntheta)
    phi_values = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    
    # Create meshgrid
    theta_grid, phi_grid = np.meshgrid(theta_values, phi_values, indexing='ij')
    theta_flat = theta_grid.flatten()
    phi_flat = phi_grid.flatten()
    
    # At poles (theta ≈ 0 or π), set all phi values to 0 to avoid duplicates
    is_pole = np.isclose(theta_flat, 0) | np.isclose(theta_flat, np.pi)
    phi_flat = np.where(is_pole, 0, phi_flat)
    
    # Convert to Cartesian coordinates
    x = pos[0] + radius * np.sin(theta_flat) * np.cos(phi_flat)
    y = pos[1] + radius * np.sin(theta_flat) * np.sin(phi_flat)
    z = pos[2] + radius * np.cos(theta_flat)
    
    # Stack into points array
    pts = np.column_stack([x, y, z])
    
    # Remove duplicates (mainly from poles)
    unique_pts, unique_indices = np.unique(pts, axis=0, return_index=True)
    unique_thetas = theta_flat[unique_indices]
    unique_phis = phi_flat[unique_indices]
    
    return unique_pts, unique_thetas, unique_phis
