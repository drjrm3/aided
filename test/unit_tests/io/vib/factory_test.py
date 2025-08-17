"""
io.vib.factory test

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from pathlib import Path
from aided.io.vib.factory import detect_log_type, log_reader_factory

from ut_helper import CxTestCase


class TestDetectLogType(CxTestCase):

    def test_no_file(self):
        """Tests failure if there is no file given."""

        self.assertEqual(detect_log_type(""), "")

    def test_bad_log_file(self):
        """Tests detection of bad log file."""

        tmp_file = Path(self.tmp_dir) / "bad.log"
        tmp_file.touch()

        with self.assertRaises(ValueError):
            foo = detect_log_type(tmp_file)
            print(foo)

    def test_gaussian_log_file(self):
        """Tests detection of Gaussian log file."""

        log_type = detect_log_type(
            f"{self.test_data_dir}/msda/g09/formamide.b3lyp.6311gss.log"
        ).lower()
        self.assertEqual(log_type, "gaussian")


class TestLogReaderFactory(CxTestCase):

    def test_no_file(self):
        """Tests failure if there is no file given."""

        with self.assertRaises(ValueError):
            log_reader_factory("")

    def test_gaussian_log_reader(self):
        """Tests creation of Gaussian log reader."""

        reader = log_reader_factory(f"{self.test_data_dir}/msda/g09/formamide.b3lyp.6311gss.log")
        # Assert that reader is of type GaussianLogReader
        self.assertEqual(reader.__class__.__name__, "GaussianLogReader")
