from datetime import date, datetime
import pandas as pd
from eurybia.core.dataset_analysis import DatasetAnalysis
import pytest


def test_ignore_cols():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    da = DatasetAnalysis(df, df, ignored_cols=["b"])
    assert "a" in da._df_test.columns
    assert "b" not in da._df_test.columns
    assert "a" in da._df_baseline.columns
    assert "b" not in da._df_baseline.columns


def test_intersecting_cols():
    df_baseline = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    df_test = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["b", "c", "d"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da.intersecting_columns == ["b", "c"]


def test_new_cols():
    df_baseline = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    df_test = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da.new_columns == ["c"]


def test_removed_cols():
    df_baseline = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    df_test = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da.removed_columns == ["c"]


def test_has_same_dtype():
    df_baseline = pd.DataFrame([[1, "2"], [4, "5"]], columns=["a", "b"])
    df_test = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da._has_same_dtype("a") is True
    assert da._has_same_dtype("b") is False


def test_dtype_mismatches():
    df_baseline = pd.DataFrame([[1, "2"], [4, "5"]], columns=["a", "b"])
    df_test = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da.dtype_mismatches == {"b": ("int64", "object")}


def test_object_cols():
    df_baseline = pd.DataFrame([[1, "2", "3"], [4, "5", "6"]], columns=["a", "b", "c"])
    df_test = pd.DataFrame([[1, "2", "2"], [3, "4", "5"]], columns=["a", "b", "d"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da.object_cols == ["b"]


def test_categorical_value_differences():
    df_baseline = pd.DataFrame([[1, "val_1", "val_4"], [4, "val_2", "val_5"]], columns=["a", "b", "c"])
    df_test = pd.DataFrame([[1, "val_3", "val_4"], [3, "val_1", "val_6"]], columns=["a", "b", "c"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da.categorical_value_differences == {
        "b": {
            "New values": ["val_3"],
            "Missing values": ["val_2"],
        },
        "c": {
            "New values": ["val_6"],
            "Missing values": ["val_5"],
        },
    }


def test_float_value_precision():
    df_test = pd.DataFrame([[1, 2.1, 3.1], [3, 5.4001, 6.1]], columns=["a", "b", "c"])
    df_baseline = pd.DataFrame([[1, 2.1, 3.11], [4, 5.4, 6.1]], columns=["a", "b", "c"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    assert da.float_precision_differences == {"b": (4, 1), "c": (1, 2)}


# def test_fix_same_cols():
#     df_test = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
#     df_baseline = pd.DataFrame([[1, 2, 3], [1, 2, 3]], columns=["b", "c", "d"])
#     da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
#     da._fix_same_cols()
#     assert list(da.df_baseline.columns) == ["b", "c"]
#     assert list(da.df_test.columns) == ["b", "c"]


def test_fix_str_fillna():
    df_test = pd.DataFrame([[1, "2", "3"], [1, None, "3"]], columns=["a", "b", "c"])
    df_baseline = pd.DataFrame([[1, "2", "3"], [1, "2", None]], columns=["a", "b", "c"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    df_test = da._fix_str_fillna(df_test)
    df_baseline = da._fix_str_fillna(df_baseline)
    assert df_test["b"].compare(pd.Series(["2", "NA"])).empty
    assert df_baseline["c"].compare(pd.Series(["3", "NA"])).empty


def test_fix_date_split():
    df_test = pd.DataFrame([pd.date_range(start="01/01/2022", end="01/01/2022")], columns=["a"])
    df_baseline = pd.DataFrame([pd.date_range(start="01/01/2022", end="01/01/2022")], columns=["a"])
    da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test)
    df_test = da._fix_date_split(df_test)
    df_baseline = da._fix_date_split(df_baseline)
    assert list(df_test.columns) == ["a_year", "a_month", "a_day"]
    assert list(df_baseline.columns) == ["a_year", "a_month", "a_day"]


# def test_fix_float_precision():
#     df_test = pd.DataFrame([[1.23, 2.345], [1.34, 2.456]], columns=["a", "b"])
#     df_baseline = pd.DataFrame([[1.234, 2.5], [1.456, 2.6]], columns=["a", "b"])
#     da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test, optional_fixes=["float_precision"])
#     da._fix_float_precision()
#     assert da.df_test["a"].compare(pd.Series([1.23, 1.34])).empty
#     assert da.df_test["b"].compare(pd.Series([2.3, 2.5])).empty
#     assert da.df_baseline["a"].compare(pd.Series([1.23, 1.46])).empty
#     assert da.df_baseline["b"].compare(pd.Series([2.5, 2.6])).empty


# def test_fix_float_to_int():
#     df_test = pd.DataFrame([[1.23, 2.345], [1.34, 2.456]], columns=["a", "b"])
#     df_baseline = pd.DataFrame([[1.23, 2], [1.45, 2]], columns=["a", "b"])
#     da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test, optional_fixes=["float_to_int"])
#     da._fix_float_to_int()
#     assert da.df_test["a"].compare(pd.Series([1.23, 1.34])).empty
#     assert da.df_test["b"].compare(pd.Series([2, 2])).empty
#     assert da.df_baseline["a"].compare(pd.Series([1.23, 1.45])).empty
#     assert da.df_baseline["b"].compare(pd.Series([2, 2])).empty


# def test_wrong_fix():
#     df_test = pd.DataFrame([[1.23, 2.345], [1.34, 2.456]], columns=["a", "b"])
#     df_baseline = pd.DataFrame([[1.23, 2], [1.45, 2]], columns=["a", "b"])
#     da = DatasetAnalysis(df_baseline=df_baseline, df_test=df_test, optional_fixes=["foo"])
#     with pytest.raises(ValueError):
#         da.fix_datasets()
