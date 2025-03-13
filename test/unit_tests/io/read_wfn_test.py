"""read_wfn test module"""

import os

from ..helper import CxTestCase, get_wfn_file

from aided.io.read_wfn import read_wfn_files, read_wfn_file

NUM_ITERS = 100
NUM_FILES = 10


class TestEDRepEquivalence(CxTestCase):
    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()

        self.input_file = self.tmp_dir + "/formamide.tst"
        with open(self.input_file, "w") as fout:
            for _ in range(NUM_FILES):
                print(self.wfn_file, file=fout)


class TestReadWfnsMulticore(CxTestCase):
    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()

        self.input_file = self.tmp_dir + "/formamide.tst"
        with open(self.input_file, "w") as fout:
            for _ in range(NUM_FILES):
                print(self.wfn_file, file=fout)

    def test_edrep_equivalence(self):
        """Tests the equivalence of EDReps."""
        wfn_rep1 = read_wfn_file(get_wfn_file())
        wfn_rep2 = read_wfn_file(get_wfn_file())
        wfn_rep3 = read_wfn_file(get_wfn_file(1))
        wfns_rep = read_wfn_files([get_wfn_file()] * NUM_FILES)

        self.assertFalse(wfn_rep1 == 1)
        self.assertTrue(wfn_rep1 == wfn_rep2)
        self.assertFalse(wfn_rep1 == wfn_rep3)
        self.assertFalse(wfn_rep1 == wfns_rep)

    def test_parallel(self):
        """Test parallel reading of wfn files."""
        wfns = [self.wfn_file] * NUM_FILES
        wfn_single_core = read_wfn_files(wfns)
        wfn_multi_core = read_wfn_files(wfns, nprocs=2)

        self.assertTrue(wfn_single_core == wfn_multi_core)
