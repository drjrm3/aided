"""read_wfn test module"""

from aided.io.read_wfn import read_wfn_files, read_wfn_file

from ut_helper import CxTestCase, get_wfn_file

NUM_ITERS = 100
NUM_FILES = 10


class ReadWfnsMulticore(CxTestCase):
    """Tests the reading of wavefunction files in multicore more."""
    def set_up(self):
        """Set up the test case."""
        self.wfn_file = get_wfn_file()

        self.input_file = self.tmp_dir + "/formamide.tst"
        with open(self.input_file, "w") as fout:
            for _ in range(NUM_FILES):
                print(self.wfn_file, file=fout)

    def test_edwfn_equivalence(self):
        """Tests the equivalence of EDWfns."""
        wfn_record1 = read_wfn_file(get_wfn_file())
        wfn_record2 = read_wfn_file(get_wfn_file())
        wfn_record3 = read_wfn_file(get_wfn_file(1))
        wfn_records = read_wfn_files([get_wfn_file()] * NUM_FILES)

        self.assertFalse(wfn_record1 == 1)
        self.assertTrue(wfn_record1 == wfn_record2)
        self.assertFalse(wfn_record1 == wfn_record3)
        self.assertFalse(wfn_record1 == wfn_records)

    def test_parallel(self):
        """Test parallel reading of wfn files."""
        wfns = [self.wfn_file] * NUM_FILES
        wfn_single_core = read_wfn_files(wfns)
        wfn_multi_core = read_wfn_files(wfns, nprocs=2)

        self.assertTrue(wfn_single_core == wfn_multi_core)
