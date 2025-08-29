"""
Static Electron Density test module

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from numpy.random import randint

from aided.core.ed.wfn_static import EDWfnStatic
from aided.core.units import Units

from ut_helper import CxTestCase, equal, get_wfn_file, read_validation_file

from conftest import STATIC_VALIDATION_FILE

NUM_ITERS = 100

# pylint: disable=protected-access


class StaticEDWfnBaseTests(CxTestCase):
    """Tests all implemented methods."""

    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()
        self.edwfn = EDWfnStatic(self.wfn_file)

    def test_units(self):
        """Test that the units are in atomic units (bohr)."""
        self.assertEqual(self.edwfn.units, Units.BOHR)

    def test_in_au(self):
        """Test that the units are in atomic units."""
        self.assertEqual(self.edwfn.in_au, True)


class StaticValidationSet(CxTestCase):
    """Tests that static electron density validation file."""

    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()

        # Read validation set.
        self.xyz, self.rho_gt, self.grad_gt, self.hess_gt = read_validation_file(
            STATIC_VALIDATION_FILE
        )

    def test_skip_double_gen_gs(self):
        """Tests that if we generate chi on the same point, it skips."""
        self.edwfn = EDWfnStatic(self.wfn_file)

        self.assertTrue(self.edwfn._gen_gs(0.0, 0.0, 0.0, 1))
        self.assertFalse(self.edwfn._gen_gs(0.0, 0.0, 0.0, 1))

    def test_0rho_validation(self):
        """Randomly tests rho values for the validation set."""
        self.edwfn = EDWfnStatic(self.wfn_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = int(randint(0, len(self.xyz) - 1))
            x, y, z = self.xyz[i]
            r = self.rho_gt[i]

            self.assertTrue(equal(r, self.edwfn.rho(x, y, z), tol=1e-12))

    def test_1grad_validation(self):
        """Randomly tests grad values for the validation set."""
        self.edwfn = EDWfnStatic(self.wfn_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = int(randint(0, len(self.xyz) - 1))
            x, y, z = self.xyz[i]
            _gx, _gy, _gz = self.grad_gt[i]

            gx, gy, gz = self.edwfn.grad(x, y, z)

            self.assertTrue(equal(gx, _gx, tol=1e-10), f"{gx} != {_gx}")
            self.assertTrue(equal(gy, _gy, tol=1e-10), f"{gy} != {_gy}")
            self.assertTrue(equal(gz, _gz, tol=1e-10), f"{gz} != {_gz}")

    def test_2hess_validation(self):
        """Randomly tests hess values for the validation set."""
        self.edwfn = EDWfnStatic(self.wfn_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = int(randint(0, len(self.xyz) - 1))
            x, y, z = self.xyz[i]
            _hxx, _hyy, _hzz, _hxy, _hxz, _hyz = self.hess_gt[i]

            hxx, hyy, hzz, hxy, hxz, hyz = self.edwfn.hess(x, y, z)

            self.assertTrue(equal(hxx, _hxx, tol=1e-10), f"{hxx} != {_hxx}")
            self.assertTrue(equal(hyy, _hyy, tol=1e-10), f"{hyy} != {_hyy}")
            self.assertTrue(equal(hzz, _hzz, tol=1e-10), f"{hzz} != {_hzz}")
            self.assertTrue(equal(hxy, _hxy, tol=1e-10), f"{hxy} != {_hxy}")
            self.assertTrue(equal(hxz, _hxz, tol=1e-10), f"{hxz} != {_hxz}")
            self.assertTrue(equal(hyz, _hyz, tol=1e-10), f"{hyz} != {_hyz}")
