"""
Helper tools for unit tests.

Copyright (C) 2025, J. Robert Michael, PhD. All Rights Reserved.
"""

import os
from pathlib import Path
import shutil
import tempfile
from typing import List, Tuple

import unittest as ut

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

    def tearDown(self):
        """Global tearDown method which children will run and call local tear_down methods."""
        shutil.rmtree(self.tmp_dir)

        self.tear_down()

    def tear_down(self):
        """Local overrideable tear_down method to each TestCase"""

    def assert_block_diagonal_msda_entries(self, msda: npt.NDArray, natoms: int):
        """Validates that the given MSDA has block diagonal entries and is proper rank."""
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

def read_validation_file(validation_file: Path) -> Tuple[List, List, List, List]:
    """Reads a validation file and returns:

        - List of xyz positions.
        - Rho scalars.
        - Gradient vectors (r_x, r_y, r_z).
        - Hessian vectors (h_xx, h_yy, h_zz, h_xy, h_xz, h_yz).
    """

    xyz = []
    rho_gt = []
    grad_gt = []
    hess_gt = []

    with open(validation_file, "r") as finp:
        for line in finp:
            if line.strip() == "" or "x y z" in line:
                continue
            x, y, z, r, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz = [
                float(x) for x in line.split()
            ]

            xyz.append([x, y, z])
            rho_gt.append(r)
            grad_gt.append([gx, gy, gz])
            hess_gt.append([hxx, hyy, hzz, hxy, hxz, hyz])

    return xyz, rho_gt, grad_gt, hess_gt

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
