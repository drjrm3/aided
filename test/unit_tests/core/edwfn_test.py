"""
edwfn and edrep test module

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from aided.core.edwfn import EDWfn
from aided.core.units import Units

from ..helper import CxTestCase, get_wfn_file

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

    def test_rho_not_implemented(self):
        """Test rho method."""
        self.edwfn = EDWfn(self.wfn_file)
        with self.assertRaises(NotImplementedError):
            self.edwfn.rho(1, 2, 3)

    def test_grad_not_implemented(self):
        """Test grad method."""
        self.edwfn = EDWfn(self.wfn_file)
        with self.assertRaises(NotImplementedError):
            self.edwfn.grad(1, 2, 3)

    def test_hess_not_implemented(self):
        """Test hess method."""
        self.edwfn = EDWfn(self.wfn_file)
        with self.assertRaises(NotImplementedError):
            self.edwfn.hess(1, 2, 3)


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
