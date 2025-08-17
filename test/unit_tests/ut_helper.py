"""
Helper tools for unit tests.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import os
import shutil
import tempfile
import unittest as ut

from conftest import VALIDATION_FILE

import numpy as np
import numpy.typing as npt


class CxTestCase(ut.TestCase):
    """High level Connex TestCase class which others can inherit from."""

    @property
    def test_data_dir(self) -> str:
        """Returns the path to the test data directory."""
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _test_data_dir = os.path.realpath(os.path.join(_this_dir, "../data"))
        if not os.path.exists(_test_data_dir):
            raise FileNotFoundError(f"Test data directory {_test_data_dir} does not exist.")
        return _test_data_dir

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

def spherical_angles_from_vector(v: npt.NDArray) -> tuple[float, float]:
    """Convert a 3D vector to spherical angles.

    Args:
        v: A 3D vector of x, y, z coordinates.

    Returns:
        theta: The polar angle (angle from the z-axis).
        phi: The azimuthal angle (angle in the x-y plane from the x-axis).
    """
    x, y, z = v
    r = np.linalg.norm(v)
    # 0 <= theta <= pi
    theta = np.arccos(z / r)
    # 0 <= phi < 2pi
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi
    return theta, phi


def equal(a, b, *, tol=1e-12) -> bool:
    """Tests if two numbers are equal within a tolerance."""
    if a == b:
        return True
    
    if abs(a) < tol or abs(b) < tol:
        # Use absolute tolerance when comparing to zero or very small numbers.
        is_equal = abs(a - b) < tol
    else:
        # Use relative tolerance for larger numbers
        is_equal = abs(a - b) / max(abs(a), abs(b)) < tol

    if not is_equal:
        print(f"[*] {a} != {b}")
        if abs(a) < tol or abs(b) < tol:
            print(f"[*] {abs(a - b)} > {tol} (absolute)")
        else:
            print(f"[*] {abs(a - b) / max(abs(a), abs(b))} > {tol} (relative)")
    
    return is_equal


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

def get_validation_file():
    return VALIDATION_FILE

