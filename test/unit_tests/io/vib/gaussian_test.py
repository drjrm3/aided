"""
io.vib.gaussian test

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy as np

from aided.io.vib.gaussian import GaussianLogReader

from ut_helper import CxTestCase
from ut_reference import GaussianLogReaderValues as GLRV


class GaussianLogReaderTest(CxTestCase):
    """Gaussian log reader tests."""

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

        self.assert_block_diagonal_msda_entries(msda, reader.natoms)
