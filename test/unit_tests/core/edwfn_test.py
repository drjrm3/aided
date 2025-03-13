"""
edwfn and edrep test module

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import os

import numpy
from numpy.random import randint

from aided.core.units import Units

from ..helper import CxTestCase, equal, get_wfn_file

from aided.core.edwfn import EDWfn

NUM_ITERS = 100


class TestEDRepNotImplemeneted(CxTestCase):
    """Tests all not implemented methods."""

    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()

    def test_read_vib_file(self):
        """Test read_vib_file method."""
        self.edwfn = EDWfn(self.wfn_file)
        with self.assertRaises(NotImplementedError):
            self.edwfn.read_vib_file("vib.tst")

    def test_read_msda_matrix(self):
        """Test read_msda_file method."""
        self.edwfn = EDWfn(self.wfn_file)
        with self.assertRaises(NotImplementedError):
            self.edwfn.read_msda_matrix("msda.tst")


class TestEDRep(CxTestCase):
    """Tests all implemented methods."""

    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()
        self.edwfn = EDWfn(self.wfn_file)

    def test_units(self):
        """Test that the units are in atomic units (bohr)."""
        self.assertEqual(self.edwfn.units, Units.BOHR)

    def test_in_au(self):
        """Test that the units are in atomic units."""
        self.assertEqual(self.edwfn.in_au, True)


class TestBCPs(CxTestCase):
    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()

    def test_find_bcps(self):
        """Test ability to find BCPs."""
        self.edrep = EDWfn(self.wfn_file)

        bcps, bond_pairs = self.edrep.find_bcps()

        self.assertEqual(len(bcps), 5)
        self.assertEqual(len(bond_pairs), 5)

        bond_pairs_gs = [
            ("C2", "H1"),
            ("C2", "N3"),
            ("C2", "O6"),
            ("H4", "N3"),
            ("H5", "N3"),
        ]

        bcp_coords_gs = numpy.array(
            [
                [-5.71710321e-01, 2.01554829e00, 0.0],
                [-6.53096058e-01, 1.01393570e-01, 0.0],
                [7.74549759e-01, 6.64836100e-01, 0.0],
                [-1.35315695e00, -2.41243187e00, 0.0],
                [-3.13186532e00, -7.87460001e-01, 0.0],
            ]
        )

        for bond_pair_at, bond_pair_gs in zip(bond_pairs, bond_pairs_gs):
            self.assertEqual(bond_pair_at, bond_pair_gs)

        for bcp_at, bcp_gs in zip(bcps, bcp_coords_gs):
            distance = numpy.linalg.norm(bcp_at.x - bcp_gs)
            self.assertTrue(distance < 1e-7)


class TestValidationSet(CxTestCase):

    def set_up(self):
        """Set up the test case."""
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        self.wfn_file = get_wfn_file()

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
                x, y, z, r, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz = [
                    float(x) for x in line.split()
                ]

                self.xyz.append([x, y, z])
                self.rho.append(r)
                self.grad.append([gx, gy, gz])
                self.hess.append([hxx, hyy, hzz, hxy, hxz, hyz])

    def test_skip_double_gen_chi(self):
        """Tests that if we generate chi on the same point, it skips."""
        self.edwfn = EDWfn(self.wfn_file)

        self.assertTrue(self.edwfn._gen_chi(0.0, 0.0, 0.0, 1))
        self.assertFalse(self.edwfn._gen_chi(0.0, 0.0, 0.0, 1))

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
