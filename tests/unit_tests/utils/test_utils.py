"""
Unit test utils.py
"""
import os
import unittest
from pathlib import Path

import pandas as pd

from eurybia.utils.utils import base_100, convert_string_to_int_keys, get_project_root, round_to_k, truncate_str


class Testutils(unittest.TestCase):
    """
    Unit test utils.py
    """

    def test_base_100(self):
        """
        Unit test base_100 function
        """
        seriestest = pd.Series([2, 8])
        resulttest = base_100(seriestest)

        assert isinstance(resulttest, pd.Series)
        pd.testing.assert_series_equal(resulttest, pd.Series([0.2, 0.8]))

    def test_truncate_str_1(self):
        """
        unit test truncate_str()
        """
        trunc_res = truncate_str(12)
        assert trunc_res == 12

    def test_truncate_str_2(self):
        """
        unit test truncate_str()
        """
        trunc_res = truncate_str("this is a test", 50)
        assert trunc_res == "this is a test"

    def test_truncate_str_3(self):
        """
        unit test truncate_str()
        """
        trunc_res = truncate_str("this is a test", 10)
        assert trunc_res == "this is a..."

    def test_convert_string_to_int_keys(self):
        """
        unit test convert_string_to_int_keys()
        """
        res = convert_string_to_int_keys({"0": 0, "1": 1})
        assert all(isinstance(k, int) for k in res)

    def test_convert_string_to_int_keys_invalid(self):
        """
        unit test convert_string_to_int_keys()
        """
        self.assertRaises(ValueError, convert_string_to_int_keys, {"foo": 0})

    def test_get_project_root_samedir(self):
        """
        unit test get_project_root()
        tests that get_project_root returns the same directory whether it is called from here or one directory up
        """
        project_root1 = get_project_root()
        test_dir = Path(__file__).parent.resolve()
        os.chdir("../")
        assert get_project_root() == project_root1
        os.chdir(test_dir)

    def test_round_to_k(self):
        """
        unit test round_to_k()
        """
        res1 = round_to_k(2.22, 2)
        res2 = round_to_k(2.22222222, 2)
        res3 = round_to_k(2.22222222, 5)
        res4 = round_to_k(2.2, 1)
        res5 = round_to_k(3, 1)
        res6 = round_to_k(3.0, 1)
        res7 = round_to_k(3.0000, 5)
        should_be_floats = [res1, res2, res3]
        should_be_ints = [res4, res5, res6, res7]

        assert len(str(res1)) == 3  # the dot counts, hence 2 figures + 1 dot = 3 chars
        assert len(str(res2)) == 3
        assert len(str(res3)) == 6  # the dot counts, hence 5 figures + 1 dot = 6 chars
        assert len(str(res4)) == 1
        assert len(str(res5)) == 1
        assert len(str(res6)) == 1
        assert len(str(res7)) == 1

        assert all(isinstance(r, float) for r in should_be_floats)
        assert all(isinstance(r, int) for r in should_be_ints)
