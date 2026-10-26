"""
Module Unit test of common.py
"""

import tempfile
import unittest
from os import path
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from eurybia.report.common import (
    VarType,
    compute_col_types,
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

    def test_series_dtype_categorical_and_unsupported(self):
        """Recognize pandas categorical types and reject dates."""
        categorical = pd.Series(pd.Categorical(["a", "b", "a"]))
        unsupported = pd.Series(pd.date_range("2024-01-01", periods=3))

        assert series_dtype(categorical) == VarType.TYPE_CAT
        assert series_dtype(unsupported) == VarType.TYPE_UNSUPPORTED

    def test_var_type_string(self):
        """Render the enum's descriptive value."""
        assert str(VarType.TYPE_NUM) == "Numeric"

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

    def test_numeric_is_continuous_threshold_and_missing_values(self):
        """Apply the unique-value threshold while ignoring missing values."""
        assert numeric_is_continuous(pd.Series(range(15))) is False
        assert numeric_is_continuous(pd.Series([*range(16), np.nan])) is True

    def test_get_callable(self):
        """
        unit test get_callable method
        """
        func = get_callable("sklearn.metrics.accuracy_score")
        y_true = [1, 1, 0, 1, 0]
        y_pred = [1, 1, 1, 0, 0]

        assert accuracy_score(y_true, y_pred) == func(y_true, y_pred)

    def test_get_callable_class_and_errors(self):
        """Resolve a class and surface invalid callable paths."""
        assert get_callable("builtins.str") is str

        with self.assertRaisesRegex(ImportError, "Empty path"):
            get_callable("")
        with self.assertRaisesRegex(ImportError, "Encountered error"):
            get_callable("builtins.nonexistent")
        with self.assertRaisesRegex(ValueError, "Invalid type"):
            get_callable("builtins.__name__")

    def test_compute_col_types(self):
        """Classify supported dataframe columns and handle missing input."""
        df = pd.DataFrame(
            {
                "category": pd.Categorical(["a", "b"]),
                "number": [1.0, 2.0],
                "date": pd.date_range("2024-01-01", periods=2),
            }
        )

        assert compute_col_types(None) == {}
        assert compute_col_types(df) == {
            "category": VarType.TYPE_CAT,
            "number": VarType.TYPE_CAT,
            "date": VarType.TYPE_UNSUPPORTED,
        }

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
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = path.join(tmp_dir, "missing.pkl")
            assert load_saved_df(path=missing_path) is None

    def test_load_saved_df_2(self):
        """
        Unit test of load_saved_df's method
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            y_path = Path(tmp_dir) / "test.pkl"
            data_y = pd.DataFrame({"test": [1, 2, 1]})
            data_y.to_pickle(y_path)

            file = load_saved_df(path=str(y_path))

            assert data_y.equals(file)
