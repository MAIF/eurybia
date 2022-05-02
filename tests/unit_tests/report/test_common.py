"""
Module Unit test of common.py
"""
import os
import unittest
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from eurybia.report.common import (
    VarType,
    display_value,
    get_callable,
    load_saved_df,
    numeric_is_continuous,
    replace_dict_values,
    series_dtype,
)


class TestCommon(unittest.TestCase):
    """
    Unit test common.py
    """

    def test_series_dtype_1(self):
        """
        Test string series
        """
        serie = pd.Series(["a", "b", "c", "d", "e", np.nan])

        assert series_dtype(serie) == VarType.TYPE_CAT

    def test_series_dtype_2(self):
        """
        Test bool series
        """
        serie = pd.Series([True, True, False, False, False])

        assert series_dtype(serie) == VarType.TYPE_CAT

    def test_series_dtype_3(self):
        """
        Test int and continuous series
        """
        serie = pd.Series(list(range(50)))

        assert series_dtype(serie) == VarType.TYPE_NUM

    def test_series_dtype_4(self):
        """
        Test float and continuous series
        """
        serie = pd.Series(np.linspace(0, 3, 50))

        assert series_dtype(serie) == VarType.TYPE_NUM

    def test_series_dtype_int_5(self):
        """
        Test int and categorical series
        """
        serie = pd.Series([1, 1, 1, 2, 2, 2])

        assert series_dtype(serie) == VarType.TYPE_CAT

    def test_series_dtype_int_6(self):
        """
        Test float and categorical series
        """
        serie = pd.Series([0.2, 0.2, 0.2, 0.6, 0.6, 0.6])

        assert series_dtype(serie) == VarType.TYPE_CAT

    def test_numeric_is_continuous_1(self):
        """
        Test int and continuous series
        """
        serie = pd.Series(list(range(50)))

        assert numeric_is_continuous(serie) is True

    def test_numeric_is_continuous_2(self):
        """
        Test float and continuous series
        """
        serie = pd.Series(np.linspace(0, 1, 100))

        assert numeric_is_continuous(serie) is True

    def test_numeric_is_continuous_3(self):
        """
        Test int and categorical series
        """
        serie = pd.Series([1, 1, 1, 2, 2, 2])

        assert numeric_is_continuous(serie) is False

    def test_numeric_is_continuous_4(self):
        """
        Test float and categorical series
        """
        serie = pd.Series([0.2, 0.2, 0.2, 0.6, 0.6, 0.6])

        assert numeric_is_continuous(serie) is False

    def test_get_callable(self):
        """
        unit test get_callable method
        """
        func = get_callable("sklearn.metrics.accuracy_score")
        y_true = [1, 1, 0, 1, 0]
        y_pred = [1, 1, 1, 0, 0]

        assert accuracy_score(y_true, y_pred) == func(y_true, y_pred)

    def test_display_value_1(self):
        """
        Unit test 1 display_value
        """
        value = 123456.789
        expected_str = "123,456.789"
        assert display_value(value, ",", ".") == expected_str

    def test_display_value_2(self):
        """
        Unit test 2 display_value
        """
        value = 123456.789
        expected_str = "123 456,789"
        assert display_value(value, " ", ",") == expected_str

    def test_display_value_3(self):
        """
        Unit test 3 display_value
        """
        value = 123456.789
        expected_str = "123.456,789"
        assert display_value(value, ".", ",") == expected_str

    def test_replace_dict_values_1(self):
        """
        Unit test 1 replace_dict_values
        """
        dico = {"a": 1234, "b": 0.1234, "c": {"d": 123.456, "e": 1234, "f": {"g": 1234}}}
        expected_d = {"a": "1,234", "b": "0.1234", "c": {"d": "123.456", "e": "1,234", "f": {"g": "1,234"}}}
        res_d = replace_dict_values(dico, display_value, ",", ".")
        assert dico == expected_d
        assert res_d == expected_d

    def test_load_saved_df_1(self):
        """
        Unit test of load_saved_df's method
        """
        file = load_saved_df(path="test")
        assert file is None

    def test_load_saved_df_2(self):
        """
        Unit test of load_saved_df's method
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent.parent
        y_path = path.join(script_path, "tests/data/test.pkl")
        data_y = pd.DataFrame({"test": [1, 2, 1]})
        data_y.to_pickle(y_path)
        file = load_saved_df(path=y_path)
        assert data_y.equals(file)
        os.remove(y_path)
