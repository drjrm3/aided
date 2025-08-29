"""
Full validation of:

    1. Reading the Gaussian log file.
    2. Reading the reference .msda file.
    3. Ensuring accuracy of the read data.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import unittest

from aided.io.vib.gaussian import GaussianLogReader

from aided.io.vib.reader import read_msda
from aided.io.vib.writer import msda_to_file

from ut_helper import CxTestCase

from conftest import TEST_DATA_DIR


class FullValidation(CxTestCase):
    """Perform full validation of reading and comparing data."""

    @unittest.skip("This is failing ... See https://github.com/drjrm3/aided/issues/70")
    def test_full_validation(self):
        """Tests the full ability to read from the log file, read from the saved GT and compare."""
        log_file = TEST_DATA_DIR / "msda/g09/formamide.b3lyp.6311gss.log"
        msda_file = TEST_DATA_DIR / "msda/g09/formamide.b3lyp.6311gss.msda"

        log_msda = GaussianLogReader(log_file).gen_msda(27)
        gt_msda = read_msda(msda_file)

        diff = log_msda - gt_msda

        print(diff / gt_msda)

        for log_row, gt_row in zip(log_msda, gt_msda):
            for log_val, gt_val in zip(log_row, gt_row):
                if log_val == 0 and gt_val == 0:
                    continue
                print(
                    f"{log_val:+20.15e} {gt_val:+20.15e} {log_val - gt_val:+20.15e} "
                    f"{(log_val - gt_val) / gt_val:+20.15e}"
                )

        atomic_numbers = [1, 6, 7, 1, 1, 8]

        msda_to_file(log_msda, atomic_numbers, "log.msda")

        # Compare the two
        self.fail()
