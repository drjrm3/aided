"""
io.vib.gaussian test

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy as np

from aided.io.vib.gaussian import GaussianLogReader

from ut_helper import CxTestCase
from ut_reference import GaussianLogReaderValues as GLRV


class TestGaussianLogReader(CxTestCase):

    def test_no_file(self):
        """Tests failure if there is no file given."""

        with self.assertRaises(FileNotFoundError):
            GaussianLogReader("non_existent_file.log")

    def test_phd_log(self):
        """Tests known ground truth from PhD work's log file."""
        log_file = f"{self.test_data_dir}/msda/g09/formamide.b3lyp.6311gss.log"

        reader = GaussianLogReader(log_file)

        self.assertEqual(reader.natoms, 6)
        self.assertEqual(reader.nmodes, 12)

        gt = GLRV()

        # Compare against ground truth.
        np.allclose(reader.modes, gt.modes, rtol=1e-12, atol=0)
        np.allclose(reader.freqs, gt.freqs, rtol=1e-12, atol=0)
        np.allclose(reader.ir_intensities, gt.ir_intensities, rtol=1e-12, atol=0)
        np.allclose(reader.reduced_masses, gt.reduced_masses, rtol=1e-12, atol=0)
        np.allclose(reader.force_constants, gt.force_constants, rtol=1e-12, atol=0)

    def test_gen_msda(self):
        """Tests generation of MSDA from known ground truth log."""
        msda_file = f"{self.test_data_dir}/msda/g09/formamide.b3lyp.6311gss.log"

        reader = GaussianLogReader(msda_file)

        msda = reader.gen_msda(27)

        # Ensure that each 3x3 block is of the form:
        # [ X, X, 0]
        # [ X, X, 0]
        # [ 0, 0, X]
        # Where X is a non-zero value.
        for i in range(reader.natoms):
            for j in range(i, reader.natoms):
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
        self.assertEqual(rank, 3 * reader.natoms - 6)
