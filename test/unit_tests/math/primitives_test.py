"""Primitives test"""

from ..helper import CxTestCase

from aided.math.primitives import gpow

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
                gt = X ** N
                # FIXME: If this always works then why do we need gpow?

                self.assertLess(abs((at - gt) / gt), 1e-14)
