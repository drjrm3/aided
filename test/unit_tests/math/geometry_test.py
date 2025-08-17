"""Geometry tests

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy as np

from ut_helper import CxTestCase, equal

from aided.math.geometry import distance_from_point_to_line, generate_spherical_grid


class DistanceFromPointToLine(CxTestCase):
    """Test distance from point to line function.


    x is the point in question, a and b define the line.
    """

    def test_distance_zero(self):
        """Test getting the distance from a point to a line.

        The point is on the line, so the distance should be zero.
        """

        for _ in range(100):
            a = np.random.uniform(-100, 100, (3, 1))
            b = np.random.uniform(-100, 100, (3, 1))

            x = a + np.random.uniform(0, 1) * (b - a)

            is_equal = equal(distance_from_point_to_line(x, a, b), 0.0, tol=1e-12)

            if not is_equal:
                print(f"a={a.T}, b={b.T}, x={x.T}, d={distance_from_point_to_line(x, a, b)}")

            self.assertTrue(is_equal)

    def test_distance_one(self):
        """Test getting the distance from a point to a line and is only one unit away."""

        for _ in range(100):
            a = np.random.uniform(-100, 100, (3, 1))
            b = np.random.uniform(-100, 100, (3, 1))

            ab = b - a
            ab /= np.linalg.norm(ab)

            # Get a perpendicular vector to ab
            if ab[0] != 0 or ab[1] != 0:
                perp = np.array([[-ab[1, 0]], [ab[0, 0]], [0]])
            else:
                perp = np.array([[0], [-ab[2, 0]], [ab[1, 0]]])

            perp /= np.linalg.norm(perp)

            x = a + np.random.uniform(0, 1) * (b - a) + perp

            self.assertTrue(equal(distance_from_point_to_line(x, a, b), 1.0, tol=1e-12))

class GenerateSphericalGrid(CxTestCase):
    """Test generating a spherical grid of points."""

    def test_spherical_grid(self):
        """Test generating a spherical grid of points."""

        for _ in range(100):
            position = np.random.uniform(-100, 100, (3, 1))
            radius = np.random.uniform(0.1, 10)
            ntheta = np.random.randint(1, 10)
            nphi = np.random.randint(1, 10)

            points, thetas, phis = generate_spherical_grid(position, radius, ntheta, nphi)

            # Test that all points are on a sphere of the given radius.
            for point in points:
                distance = float(np.linalg.norm(point - position.flatten()))
                self.assertAlmostEqual(distance, radius)
            
            # Test that thetas are between 0 and pi
            for theta in thetas:
                self.assertGreaterEqual(theta, 0)
                self.assertLessEqual(theta, np.pi)
            
            # Test that phis are between 0 and 2*pi
            for phi in phis:
                self.assertGreaterEqual(phi, 0)
                self.assertLessEqual(phi, 2 * np.pi)

            # Test that the points are all unique.
            unique_points = np.unique(points, axis=0)  # axis=0 for rows
            self.assertEqual(len(unique_points), len(points))




