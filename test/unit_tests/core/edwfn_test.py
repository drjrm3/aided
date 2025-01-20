"""edwfn test module"""

import os

from numpy.random import randint

from ..helper import CxTestCase, equal

from aided.core.edwfn import EDWfn

NUM_ITERS = 100


class TestValidationSet(CxTestCase):

    def set_up(self):
        """Set up the test case."""
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        self.wfn_file = os.path.join(_this_dir, "..", "..", "data", "wfns", "formamide", "formamide.6311gss.b3lyp.wfn")

        # Read validation set.
        self.validation_file = os.path.join(_this_dir, "..", "..", "validation", "validation.txt")

        self.xyz = []
        self.rho = []
        self.grad = []
        self.hess = []

        with open(self.validation_file, "r") as finp:
            for line in finp:
                if line.strip() == "" or "x y z" in line:
                    continue
                x, y, z, r, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz = [float(x) for x in line.split()]

                self.xyz.append([x, y, z])
                self.rho.append(r)
                self.grad.append([gx, gy, gz])
                self.hess.append([hxx, hyy, hzz, hxy, hxz, hyz])


    def test_0rho_validation(self):
        """Randomly tests rho values for the validation set."""
        self.edwfn = EDWfn(self.wfn_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = randint(0, len(self.xyz) - 1)
            x, y, z = self.xyz[i]
            r = self.rho[i]

            self.assertTrue(equal(r, self.edwfn.rho(x, y, z), 1e-12))

    def test_1grad_validation(self):
        """Randomly tests grad values for the validation set."""
        self.edwfn = EDWfn(self.wfn_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = randint(0, len(self.xyz) - 1)
            x, y, z = self.xyz[i]
            _gx, _gy, _gz = self.grad[i]

            gx, gy, gz = self.edwfn.grad(x, y, z)

            self.assertTrue(equal(gx, _gx), 1e-12)
            self.assertTrue(equal(gy, _gy), 1e-12)
            self.assertTrue(equal(gz, _gz), 1e-12)

    def test_2hess_validation(self):
        """Randomly tests hess values for the validation set."""
        self.edwfn = EDWfn(self.wfn_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = randint(0, len(self.xyz) - 1)
            x, y, z = self.xyz[i]
            _hxx, _hyy, _hzz, _hxy, _hxz, _hyz = self.hess[i]

            hxx, hyy, hzz, hxy, hxz, hyz = self.edwfn.hess(x, y, z)

            self.assertTrue(equal(hxx, _hxx), 1e-12)
            self.assertTrue(equal(hyy, _hyy), 1e-12)
            self.assertTrue(equal(hzz, _hzz), 1e-12)
            self.assertTrue(equal(hxy, _hxy), 1e-12)
            self.assertTrue(equal(hxz, _hxz), 1e-12)
            self.assertTrue(equal(hyz, _hyz), 1e-12)
