"""Primitives test"""

from ..helper import CxTestCase

from aided.math.primitives import gpow

import numpy as np
from numpy import random


class TestGpow(CxTestCase):

    def test_gpow(self):
        """Test gpow"""

        # Ensure that anything raised to the 0 is 1.
        for x in random.random(10):
            X = -100 + 200 * x
            self.assertEqual(gpow(X, 0), 1)

        # 0 to any exponent is 0.
        for n in random.random(100):
            N = int(-10 + 20 * n)
            if N == 0:
                continue
            self.assertEqual(gpow(0, N), 0)

        # Test some random values.
        for x in random.random(100):
            X = -100 + 200 * x
            for n in random.random(100):
                N = int(-10 + 20 * n)
                at = gpow(X, N)
                if X == 0 and N < 0:
                    self.assertEqual(at, 0.0)
                    continue
                gt = X**N

                self.assertLess(abs((at - gt) / gt), 1e-14)

    """
    def test_gpow_array_x_scalar_expon(self):
        ""Test gpow where x is an array and expon is a scalar""

        x = -100 + 200 * random.random(100)  # Array of 100 values
        x[0] = 0  # Test zero base case
        expon = 3  # Scalar exponent
        results = gpow(x, expon)
        expected = x ** expon

        # Ensure zero base is handled correctly
        self.assertEqual(results[0], 0.0)

        # Check all other values
        for r, e in zip(results[1:], expected[1:]):
            self.assertLess(abs((r - e) / e), 1e-14)

    def test_gpow_scalar_x_array_expon(self):
        ""Test gpow where x is a scalar and expon is an array""

        x = 2.0  # Scalar base
        expon = np.random.randint(-10, 10, size=100)  # Array of exponents
        expon[0] = 0  # Ensure zero exponent is handled
        expon[1] = -5  # Test negative exponent case
        results = gpow(x, expon)
        expected = x ** expon

        # Ensure anything to power 0 is 1
        self.assertEqual(results[0], 1.0)

        # Ensure negative exponent case matches expected
        self.assertLess(abs((results[1] - expected[1]) / expected[1]), 1e-14)

        # Check all other values
        for r, e in zip(results[2:], expected[2:]):
            self.assertLess(abs((r - e) / e), 1e-14)
    """

    def test_gpow_array_x_array_expon(self):
        """Test gpow where both x and expon are arrays"""

        x = -100 + 200 * random.random(100)  # Array of 100 base values
        expon = np.random.randint(-10, 10, size=100)  # Array of 100 exponents
        x[0] = 0  # Ensure zero base is tested
        expon[1] = 0  # Ensure zero exponent is tested
        expon[2] = -5  # Ensure negative exponent case is tested
        results = gpow(x, expon)

        # It is expected that one of the cases will be 0**n where n may be negative.
        # We handle this as ignoring the 'expected' value though and ensure gpow gives 0.
        with np.errstate(divide="ignore", invalid="ignore"):  # Suppress divide-by-zero warnings
            expected = x**expon

        # Ensure zero base is handled correctly
        self.assertEqual(results[0], 0.0)

        # Ensure anything to power 0 is 1
        self.assertEqual(results[1], 1.0)

        # Ensure negative exponent case matches expected
        self.assertLess(abs((results[2] - expected[2]) / expected[2]), 1e-14)

        # Check all other values
        for r, e in zip(results[3:], expected[3:]):
            self.assertLess(abs((r - e) / e), 1e-14)
