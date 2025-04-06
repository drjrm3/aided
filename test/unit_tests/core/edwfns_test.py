"""edwfn test module"""

import os

from numpy.random import randint
from datetime import datetime

from ..helper import CxTestCase, equal

from aided.core.edwfns import EDWfns

NUM_ITERS = 100
NUM_FILES = 10


class TestWfns(CxTestCase):

    def set_up(self):
        """Set up the test case."""
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        self.wfn_file = os.path.join(
            _this_dir, "..", "..", "data", "wfns", "formamide", "formamide.6311gss.b3lyp.wfn"
        )

        # Create a list of wfn files to process in a file.
        self.wfn_list_file = self.tmp_dir + "/formamide.tst"
        with open(self.wfn_list_file, "w") as fout:
            for _ in range(NUM_FILES):
                print(self.wfn_file, file=fout)

        # Read validation set.
        self.validation_file = os.path.join(_this_dir, "..", "..", "validation", "validation.txt")

        self.xyz = []
        # Ground Truth values.
        self.rho_gt = []
        self.grad_gt = []
        self.hess_gt = []

        with open(self.validation_file, "r") as finp:
            for line in finp:
                if line.strip() == "" or "x y z" in line:
                    continue
                x, y, z, r, gx, gy, gz, hxx, hxy, hxz, hyy, hyz, hzz = [
                    float(x) for x in line.split()
                ]

                self.xyz.append([x, y, z])
                self.rho_gt.append(r)
                self.grad_gt.append([gx, gy, gz])
                self.hess_gt.append([hxx, hyy, hzz, hxy, hxz, hyz])

    def test_atpos(self):
        """Test the averaged atomic positions."""
        self.edwfns = EDWfns(self.wfn_list_file)

        gs_pos = [
            [-0.87278233, 2.69428921, 0.0],
            [0.0, 0.79115302, 0.0],
            [-1.76946457, -1.07580723, 0.0],
            [-1.1993322, -2.89432062, 0.0],
            [-3.62981289, -0.68496552, 0.0],
            [2.26102242, 0.45859118, 0.0],
        ]

        for at, gs in zip(self.edwfns.atpos, gs_pos):
            for a, g in zip(at, gs):
                self.assertTrue(equal(a, g))

    def test_atnames(self):
        """Test atom names."""
        self.edwfns = EDWfns(self.wfn_list_file)

        for at, gs in zip(self.edwfns.atnames, ["H1", "C2", "N3", "H4", "H5", "O6"]):
            self.assertTrue(at == gs)

    def test_no_recompute(self):
        """Tests that we don't recompute the same point."""
        self.edwfns = EDWfns(self.wfn_list_file)

        tic = datetime.now()
        rho1 = self.edwfns.rho(0.0, 0.0, 0.0)
        toc = datetime.now()
        diff1 = toc - tic

        tic = datetime.now()
        rho2 = self.edwfns.rho(0.0, 0.0, 0.0)
        toc = datetime.now()
        diff2 = toc - tic

        # Ensure the values are equal.
        self.assertTrue(rho1 == rho2)

        # Ensure the time it took for the second computaton is at least twice as fast.
        self.assertTrue(diff1 / diff2 > 2.0)

    def test_equivalence(self):
        """Tests the equivalence of two EDWfns objects."""
        self.edwfns1 = EDWfns(self.wfn_list_file)
        self.edwfns2 = EDWfns(self.wfn_list_file)

        self.assertTrue(self.edwfns1 == self.edwfns2)

    def test_0rho_validation(self):
        """Randomly tests rho values for the validation set."""

        self.edwfns = EDWfns(self.wfn_list_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = randint(0, len(self.xyz) - 1)
            x, y, z = self.xyz[i]
            rho = self.rho_gt[i]

            self.assertTrue(equal(rho, self.edwfns.rho(x, y, z), 1e-12))

    def test_1grad_validation(self):
        """Randomly tests grad values for the validation set."""
        self.edwfns = EDWfns(self.wfn_list_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = randint(0, len(self.xyz) - 1)
            x, y, z = self.xyz[i]
            _gx, _gy, _gz = self.grad_gt[i]

            gx, gy, gz = self.edwfns.grad(x, y, z)

            self.assertTrue(equal(gx, _gx), 1e-12)
            self.assertTrue(equal(gy, _gy), 1e-12)
            self.assertTrue(equal(gz, _gz), 1e-12)

    def test_2hess_validation(self):
        """Randomly tests hess values for the validation set."""
        self.edwfns = EDWfns(self.wfn_list_file)

        for _ in range(NUM_ITERS):
            # Get random integer between 0 and len(self.xyz) - 1
            i = randint(0, len(self.xyz) - 1)
            x, y, z = self.xyz[i]
            hxx_gt, hyy_gt, hzz_gt, hxy_gt, hxz_gt, hyz_gt = self.hess_gt[i]

            # Compute with and without batch size
            hxx, hyy, hzz, hxy, hxz, hyz = self.edwfns.hess(x, y, z)
            hxx_, hyy_, hzz_, hxy_, hxz_, hyz_ = self.edwfns.hess(x, y, z, batch_size=0)

            # Assert the values are equal.
            for h, h_ in zip([hxx, hyy, hzz, hxy, hxz, hyz], [hxx_, hyy_, hzz_, hxy_, hxz_, hyz_]):
                self.assertTrue(equal(h, h_), 1e-12)

            for h, h_gt in zip(
                [hxx, hyy, hzz, hxy, hxz, hyz], [hxx_gt, hyy_gt, hzz_gt, hxy_gt, hxz_gt, hyz_gt]
            ):
                self.assertTrue(equal(h, h_gt), 1e-12)
