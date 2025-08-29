"""
io.vib.reader test

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy as np

from aided.io.vib.reader import read_msda

from ut_helper import CxTestCase


class TestMSDAWriter(CxTestCase):

    natoms = 6

    def test_no_file(self):
        """Tests failure if there is no file given."""
        with self.assertRaises(FileNotFoundError):
            read_msda("no_file.msda")

    def test_read_phd_msda(self):
        """Tests reading a MSDA file from PhD ground truth data."""
        msda_file = f"{self.test_data_dir}/msda/g09/formamide.b3lyp.6311gss.msda"

        msda = read_msda(msda_file)

        natoms = 6

        # Ensure that each 3x3 block is of the form:
        # [ X, X, 0]
        # [ X, X, 0]
        # [ 0, 0, X]
        # Where X is a non-zero value.
        for i in range(natoms):
            for j in range(i, natoms):
                block = msda[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3]
                self.assertEqual(block[0, 2], 0)
                self.assertEqual(block[1, 2], 0)
                self.assertEqual(block[2, 0], 0)
                self.assertEqual(block[2, 1], 0)

                self.assertNotEqual(block[0, 0], 0)
                self.assertNotEqual(block[0, 1], 0)
                self.assertNotEqual(block[1, 0], 0)
                self.assertNotEqual(block[1, 1], 0)
                self.assertNotEqual(block[2, 2], 0)

        # Ensure that the matrix is symmetric.
        self.assertTrue(np.allclose(msda, msda.T, rtol=1e-12, atol=0))

        # Ensure the rank of the matrix is 3N-6.
        rank = np.linalg.matrix_rank(msda)
        self.assertEqual(rank, 3 * natoms - 6)

    def test_read_msda_bad_shapes(self):
        """Test reading a MSDA file with bad shapes."""

        # Test reading an MSDA with less lines than atoms.
        with open(f"{self.tmp_dir}/bad_shape.msda", "w") as f:
            f.write("H1x: 1 2 3 4 5 6")

        with self.assertRaises(ValueError):
            read_msda(f"{self.tmp_dir}/bad_shape.msda")
