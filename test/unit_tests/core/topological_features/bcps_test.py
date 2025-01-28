"""
Test functionality for BCPs.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy

from ...helper import CxTestCase, get_wfn_file

from aided.core.edwfn import EDWfn

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
