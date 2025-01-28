"""read_wfn test module"""

import os

from ..helper import CxTestCase

from aided.io.read_wfn import read_wfn_files

NUM_ITERS = 100
NUM_FILES = 10


class TestReadWfnsMulticore(CxTestCase):

    def set_up(self):
        """Set up the test case."""
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        self.wfn_file = os.path.join(_this_dir, "..", "..", "data", "wfns", "formamide", "formamide.6311gss.b3lyp.wfn")

        self.input_file = self.tmp_dir + "/formamide.tst"
        with open(self.input_file, "w") as fout:
            for _ in range(NUM_FILES):
                print(self.wfn_file, file=fout)

    def test_parallel(self):
        """Test parallel reading of wfn files."""
        wfns = [self.wfn_file] * NUM_FILES
        wfn_single_core = read_wfn_files(wfns)
        wfn_multi_core = read_wfn_files(wfns, nprocs=2)

        self.assertTrue(wfn_single_core == wfn_multi_core)
