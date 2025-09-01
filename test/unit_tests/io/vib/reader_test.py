"""
io.vib.reader test

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from aided.io.vib.reader import read_msda

from ut_helper import CxTestCase


class MsdaReader(CxTestCase):
    """Test the MSDA reader."""

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

        self.assert_block_diagonal_msda_entries(msda, natoms)

    def test_read_msda_bad_shapes(self):
        """Test reading a MSDA file with bad shapes."""

        # Test reading an MSDA with less lines than atoms.
        with open(f"{self.tmp_dir}/bad_shape.msda", "w") as f:
            f.write("H1x: 1 2 3 4 5 6")

        with self.assertRaises(ValueError):
            read_msda(f"{self.tmp_dir}/bad_shape.msda")
