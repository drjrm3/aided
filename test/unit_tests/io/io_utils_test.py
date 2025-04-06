"""
io.utils test module

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

from aided.core.edrep import EDRepType
from ..helper import CxTestCase

from aided.io import utils


class TestIOUtils(CxTestCase):

    def test_convert_scientific_notation(self):
        """Tests conversion of scientific notation from 1.0D+01 to 1.0E+01"""

        lines = ["1.0D+01 2.1D-01", "1.0D-01", "1.0D-01", "1.0D+1", "1.0D-1", "1.0D+1"]
        expected = ["1.0E+01 2.1E-01", "1.0E-01", "1.0E-01", "1.0E+1", "1.0E-1", "1.0E+1"]

        self.assertEqual(utils.convert_scientific_notation(lines), expected)

    def test_is_number(self):
        """Tests if a string is a float."""

        self.assertTrue(utils.is_number("1"))
        self.assertTrue(utils.is_number("-1"))
        self.assertTrue(utils.is_number("0"))
        self.assertTrue(utils.is_number("1.0"))
        self.assertTrue(utils.is_number("1.0E+01"))
        self.assertFalse(utils.is_number("1.0D+01"))
        self.assertFalse(utils.is_number("foo"))

    def test_get_edrep_by_file_extension(self):
        """Tests the ability to get a type of EDRep by file extension."""

        self.assertEqual(utils.get_edrep_type_by_file_extention("foo.wfn"), EDRepType.WFN)

        # Assert that exention ".foo" raises a ValueError
        with self.assertRaises(ValueError):
            utils.get_edrep_type_by_file_extention("foo.foo")
