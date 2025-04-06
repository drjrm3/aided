"""
Helper tools for unit tests.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import os
import shutil
import tempfile
import unittest as ut


class CxTestCase(ut.TestCase):
    """High level Connex TestCase class which others can inherit from."""

    def setUp(self):
        """Global setUp method which others will run and call local set_up methods."""
        self.tmp_dir = tempfile.mkdtemp()

        self.set_up()

    def set_up(self):
        """Local overrideable set_up method to each TestCase"""
        pass

    def tearDown(self):
        """Global tearDown method which children will run and call local tear_down methods."""
        shutil.rmtree(self.tmp_dir)

        self.tear_down()

    def tear_down(self):
        """Local overrideable tear_down method to each TestCase"""
        pass


def equal(a, b, tol=1e-12):
    """Tests if two numbers are equal within a tolerance."""
    if a == b:
        return True
    is_equal = abs(a - b) / a < tol

    if not is_equal:
        print(f"[*] {a} != {b}")
        print(f"[*] {abs(a - b) / a} > {tol}")
    return abs(a - b) / b < tol


def get_wfn_file(ifile: int = 0) -> str:
    """Returns the path to the test wavefunction."""
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    form_dir = os.path.join(_this_dir, "../data/wfns/formamide")
    if ifile == 0:
        return f"{form_dir}/formamide.6311gss.b3lyp.wfn"

    form_file = f"{form_dir}/form{ifile:06d}.wfn"

    if os.path.exists(form_file):
        return form_file
    raise FileNotFoundError(f"File {form_file} does not exist.")
