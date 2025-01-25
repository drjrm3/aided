"""core.utils test module"""

from ..helper import CxTestCase

from aided.core import utils


class TestSplitWork(CxTestCase):
    """Tests generated or based upon ChatGPT's recommendations."""

    def test_even_division(self):
        """Test when n is evenly divisible by num_groups."""
        self.assertEqual(utils.split_work(10, 2), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    def test_uneven_division(self):
        """Test when n is not evenly divisible by num_groups."""
        self.assertEqual(utils.split_work(10, 3), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_single_group(self):
        """Test when there is only one group."""
        self.assertEqual(utils.split_work(5, 1), [[0, 1, 2, 3, 4]])

    def test_more_groups_than_units(self):
        """Test when num_groups is greater than n."""
        self.assertEqual(utils.split_work(3, 5), [[0], [1], [2]])

    def test_no_units(self):
        """Test when there are no units (n=0)."""
        self.assertEqual(utils.split_work(0, 3), [])

    def test_no_groups(self):
        """Test when num_groups is 0, should raise an exception."""
        with self.assertRaises(ZeroDivisionError):
            utils.split_work(10, 0)

    def test_large_number_of_units(self):
        """Test with a large number of units."""
        n = 1000
        num_groups = 10
        result = utils.split_work(n, num_groups)
        self.assertEqual(len(result), num_groups)
        self.assertTrue(all(len(group) == 100 for group in result))
