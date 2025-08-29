"""
io.vib.writer test

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import numpy as np

from aided.io.vib.writer import msda_to_file

from ut_helper import CxTestCase


class TestMSDAWriter(CxTestCase):

    natoms = 6

    def test_no_file(self):
        """Tests failure if there is no file given."""
        with self.assertRaises(FileNotFoundError):
            msda = np.random.rand(self.natoms * 3, self.natoms * 3)
            msda_to_file(msda, [i + 1 for i in range(self.natoms)], "")
            raise

    def test_write_msda_bad_shapes(self):
        """Tests writing a MSDA with mismatched shapes"""
        msda_file = f"{self.tmp_dir}/test.msda"

        # Single dimension msda
        msda = np.random.rand(5)
        with self.assertRaises(ValueError):
            msda_to_file(msda, [], msda_file)

        # MSDA not multiple of 3.
        msda = np.random.rand(10, 10)
        with self.assertRaises(ValueError):
            msda_to_file(msda, [], msda_file)

        # MSDA not square.
        msda = np.random.rand(9, 12)
        with self.assertRaises(ValueError):
            msda_to_file(msda, [], msda_file)

        # Atom list not sized as 1/3 of msda.
        msda = np.random.rand(9, 9)
        with self.assertRaises(ValueError):
            msda_to_file(msda, [1, 2], msda_file)

    def test_simple_writer(self):
        """Tests writing a simple MSDA"""

        msda_file = f"{self.tmp_dir}/test.msda"

        msda = np.random.rand(self.natoms * 3, self.natoms * 3)
        atoms = [i + 1 for i in range(self.natoms)]
        msda_to_file(msda, atoms, msda_file)
