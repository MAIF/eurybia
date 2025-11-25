"""dataset_analysis module"""

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype, is_integer_dtype, is_object_dtype
from sklearn.model_selection import train_test_split

MANDATORY_FIXES = ["same_cols", "str_fillna", "numeric_fillna", "date_split"]
DEFAULT_OPTIONAL_FIXES = ["float_precision", "float_to_int"]


class DatasetAnalysis:
    """
    DatasetAnalysis provides tools for comparing and analyzing differences between two pandas DataFrames,
    typically representing a baseline and a test dataset. It supports detection and fixing of schema and data
    inconsistencies, such as column mismatches, data type differences, missing values, and float precision issues.

    Key Features:
    - Identifies intersecting, new, and removed columns between datasets.
    - Detects data type mismatches and float precision differences.
    - Analyzes categorical value differences between datasets.
    - Provides utilities for filling missing values and aligning schemas.
    - Supports splitting date columns and converting float columns to integers.
    - Offers a train/test split method that labels data for downstream analysis.
        sample_size (int | None, optional): If provided, samples up to this many rows from each DataFrame.
        ignored_cols (list[str] | None, optional): List of column names to ignore during analysis.
        optional_fixes (list[str] | None, optional): List of optional data cleaning fixes to apply.

    Attributes:
        df_baseline (pd.DataFrame): The processed baseline DataFrame.
        df_test (pd.DataFrame): The processed test DataFrame.
        df_concat (pd.DataFrame): Concatenated and labeled DataFrame for combined analysis.
        ignored_cols (list[str]): List of columns ignored in analysis.
        fixes_state (dict): Tracks which data cleaning fixes have been applied.
        valid_columns (list[str]): Columns present in both datasets with matching data types.
        intersecting_columns (list[str]): Columns present in both datasets.
        new_columns (list[str]): Columns present only in the test dataset.
        removed_columns (list[str]): Columns present only in the baseline dataset.
        dtype_mismatches (dict): Columns with differing data types between datasets.
        object_cols (list[str]): Categorical (object dtype) columns with matching types.
        float_cols (list[str]): Float columns with matching types.
        datetime_cols (list[str]): Datetime columns with matching types.
        categorical_value_differences (dict): Differences in categorical values between datasets.
        float_precision_differences (dict): Differences in float precision between datasets.
        cat_features_indices (list[int]): Indices of categorical features in the concatenated DataFrame.

    Methods:
        fix_datasets(): Applies configured data cleaning and transformation fixes to the datasets.
        train_test_split(**kwargs): Splits and labels the datasets for train/test analysis.
    """

    def __init__(
        self,
        df_test: pd.DataFrame,
        df_baseline: pd.DataFrame,
        sample_size: int | None = None,
        ignored_cols: list[str] | None = None,
        optional_fixes: list[str] | None = None,
    ):
        """
        Initializes the DatasetAnalysis class.

        Args:
            df_baseline (pd.DataFrame): The baseline DataFrame.
            df_test (pd.DataFrame): The test DataFrame to compare against the baseline.
            ignore_cols (list[str] | None): List of column names to ignore during analysis. Defaults to None.
        """
        self._ignored_cols: list[str] = ignored_cols if ignored_cols is not None else []
        self._df_test = df_test.drop(self._ignored_cols, axis=1, errors="ignore")
        self._df_baseline = df_baseline.drop(self._ignored_cols, axis=1, errors="ignore")

        if sample_size is not None:
            if self._df_test.shape[0] > sample_size:
                self._df_test = self._df_test.sample(sample_size)
            if self._df_baseline.shape[0] > sample_size:
                self._df_baseline = self._df_baseline.sample(sample_size)

        self._fixes = MANDATORY_FIXES
        if optional_fixes is None:
            optional_fixes = DEFAULT_OPTIONAL_FIXES
        self._fixes += optional_fixes

        self._fixes_state = {fix: False for fix in self._fixes}
        self._df_concat: pd.DataFrame = None

    @property
    def df_baseline(self) -> pd.DataFrame:
        """Getter"""
        return self._df_baseline

    @df_baseline.setter
    def df_baseline(self, val) -> None:
        self._df_baseline = val

    @property
    def df_test(self) -> pd.DataFrame:
        """Getter"""
        return self._df_test

    @df_test.setter
    def df_test(self, val) -> None:
        self._df_test = val

    @property
    def df_concat(self) -> pd.DataFrame:
        """Concatenates the baseline and test DataFrames and returns a single DataFrame.

        Returns:
            pd.DataFrame: A DataFrame resulting from concatenating `self.df_baseline` and `self.df_test`,
            with the index reset.
        """
        if self._df_concat is None:
            train, test = self.train_test_split(test_size=0.25, random_state=42)
            self._df_concat = pd.concat([train, test]).reset_index(drop=True)
        return self._df_concat

    @property
    def ignored_cols(self) -> list[str]:
        """Getter"""
        return self._ignored_cols

    @property
    def fixes_state(self) -> dict:
        """Getter"""
        return self._fixes_state

    def _set_fix_applied(self, fix) -> None:
        if fix not in self._fixes_state.keys():
            raise RuntimeError("")
        self._fixes_state[fix] = True

    @property
    def valid_columns(self) -> list[str]:
        """Returns the list of intersecting column names that have the same data type across datasets.

        Returns:
            list[str]: A list of valid column names with consistent data types.
        """

        return [c for c in self.intersecting_columns if self._has_same_dtype(c)]

    @property
    def intersecting_columns(self) -> list[str]:
        """Returns a list of column names that are present in both df_test and df_baseline.

        Returns:
            list[str]: List of column names common to both df_test and df_baseline.
        """
        return [c for c in self.df_test.columns if c in self.df_baseline.columns]

    @property
    def new_columns(self) -> list[str]:
        """Identifies columns that have been added by comparing df_test and df_baseline.

        Returns:
            list[str]: A list of column names that exist in df_test but not in df_baseline.
        """
        return [c for c in self.df_test.columns if c not in self.df_baseline.columns]

    @property
    def removed_columns(self) -> list[str]:
        """Identifies columns that have been removed by comparing df_baseline and df_test.

        Returns:
            list[str]: A list of column names that exist in df_baseline but not in df_test.
        """
        return [c for c in self.df_baseline.columns if c not in self.df_test.columns]

    def _has_same_dtype(self, colname: str) -> bool:
        """Check if the specified column has the same data type in both the baseline and test DataFrames.

        Args:
            colname (str): The name of the column to compare.
        Returns:
            bool: True if the column has the same data type in both DataFrames, False otherwise.
        """

        return self.df_baseline[colname].dtype == self.df_test[colname].dtype

    @property
    def dtype_mismatches(self) -> dict[str, tuple[str, str]]:
        """Identifies columns with mismatched data types between the baseline and test DataFrames.

        Returns:
            dict[str, tuple[str, str]]:
                A dictionary where each key is a column name with a data type mismatch,
                and the value is a tuple containing the data types from the baseline and test DataFrames, respectively.
        """
        dtype_mismatches: dict[str, tuple[str, str]] = dict()
        for colname in self.intersecting_columns:
            if not self._has_same_dtype(colname):
                dtype_mismatches[colname] = (
                    str(self.df_test[colname].dtype),
                    str(self.df_baseline[colname].dtype),
                )
        return dtype_mismatches

    @property
    def object_cols(self) -> list[str]:
        """Returns a list of column names from the intersecting and not dtype mismatching columns
        whose data type is string.

        Returns:
            list[str]: List of column names with 'object' dtype and no dtype mismatches.
        """

        return [colname for colname in self.valid_columns if is_object_dtype(self.df_baseline[colname].dtype)]

    @property
    def float_cols(self) -> list[str]:
        """Returns a list of column names from the intersecting and not dtype mismatching columns
        whose data type is float.

        Returns:
            list[str]: List of column names with 'float' dtype and no dtype mismatches.
        """

        return [colname for colname in self.valid_columns if is_float_dtype(self.df_baseline[colname])]

    @property
    def datetime_cols(self) -> list[str]:
        """Returns a list of column names from the intersecting and not dtype mismatching columns
        whose data type is datetime.

        Returns:
            list[str]: List of column names with datetime dtype and no dtype mismatches.
        """
        return [colname for colname in self.valid_columns if is_datetime64_any_dtype(self.df_baseline[colname])]

    @property
    def categorical_value_differences(self):
        """Identifies and returns the differences in categorical (object) column values
        between the baseline and test datasets.

        For each object column, the method compares the unique values (modalities)
        present in the baseline and test datasets.
        It records any new values found in the test dataset that are not in the baseline,
        as well as any missing values that are present in the baseline but absent in the test dataset.

        Returns:
            dict: A dictionary where each key is an object column name, and the value is another dictionary with:
                - "New values": List of values present in the test dataset but not in the baseline.
                - "Missing values": List of values present in the baseline but not in the test dataset.
        """
        different_modalities: dict[str, dict[str, list[str]]] = dict()
        for colname in self.object_cols:
            baseline_modalities = pd.unique(self.df_baseline[colname])
            test_modalities = pd.unique(self.df_test[colname])

            new_modalities = [mod for mod in test_modalities if mod not in baseline_modalities]
            missing_modalities = [mod for mod in baseline_modalities if mod not in test_modalities]

            if (len(new_modalities) > 0) or (len(missing_modalities) > 0):
                different_modalities[colname] = {
                    "New values": new_modalities,
                    "Missing values": missing_modalities,
                }
        return different_modalities

    @property
    def float_precision_differences(self):
        """Analyzes the precision (number of digits after the decimal point) of float columns
        in the baseline and test datasets.

        For each float column, calculates the maximum number of decimal digits present
        in both the baseline and test datasets. If the maximum precision differs between the two datasets
        for any column, records the column name and the respective precisions.

        Returns:
            dict: A dictionary where keys are column names with differing float precisions, and values are tuples
                  of (test dataset max precision, baseline dataset max precision).
        """

        def nb_digits(i: float):
            split_i = str(i).split(".")
            return len(split_i[1]) if len(split_i) > 1 else 0

        different_float_precision: dict[str, tuple[str, str]] = dict()

        for colname in self.float_cols:
            baseline_digits = self.df_baseline[colname].apply(nb_digits)
            test_digits = self.df_test[colname].apply(nb_digits)
            baseline_max_precision = np.max(baseline_digits)
            test_max_precision = np.max(test_digits)
            if baseline_max_precision != test_max_precision:
                different_float_precision[colname] = (test_max_precision, baseline_max_precision)

        return different_float_precision

    @property
    def cat_features_indices(self) -> list[int]:
        """Returns a list of indices corresponding to categorical features in the concatenated DataFrame.

        Iterates over the columns of `self.df_concat` and checks if each column is present in `self.object_cols`,
        which is assumed to contain the names of categorical columns. The index of each categorical column is
        appended to the returned list.

        Returns:
            list[int]: List of indices for categorical features in `self.df_concat`.
        """

        i = 0
        indice_cat = []
        for col in self.df_concat.columns:
            if col in self.object_cols:
                indice_cat.append(i)
            i += 1
        return indice_cat

    def fix_datasets(self) -> None:
        """Applies a series of data cleaning and transformation fixes to the datasets.

        The method sequentially performs the following operations:
            - Fixes columns with identical names.
            - Splits date columns if necessary.
            - Fills missing string values.
            - Applies float precision fixes if specified in `_fixes`.
            - Converts float columns to integers if specified in `_fixes`.
        The specific fixes applied depend on the configuration in the `_fixes` attribute.
        """

        self._fix_same_cols()
        self._fix_date_split()
        self._fix_str_fillna()
        self._fix_numeric_fillna()

        if "float_precision" in self._fixes:
            self._fix_float_precision()

        if "float_to_int" in self._fixes:
            self._fix_float_to_int()

    def _fix_same_cols(self) -> None:
        if not self.fixes_state["same_cols"]:
            self.df_baseline = self.df_baseline[self.intersecting_columns]
            self.df_test = self.df_test[self.intersecting_columns]
            self._set_fix_applied("same_cols")

    def _fix_date_split(self) -> None:
        if not self.fixes_state["date_split"]:
            for col in self.datetime_cols:
                self.df_baseline[col + "_year"] = self.df_baseline[col].dt.year
                self.df_baseline[col + "_month"] = self.df_baseline[col].dt.month
                self.df_baseline[col + "_day"] = self.df_baseline[col].dt.day
                self.df_baseline = self.df_baseline.drop(col, axis=1)

                self.df_test[col + "_year"] = self.df_test[col].dt.year
                self.df_test[col + "_month"] = self.df_test[col].dt.month
                self.df_test[col + "_day"] = self.df_test[col].dt.day
                self.df_test = self.df_test.drop(col, axis=1)

            self._set_fix_applied("date_split")

    def _fix_str_fillna(self) -> None:
        if not self.fixes_state["str_fillna"]:
            self.df_baseline[self.object_cols] = self.df_baseline[self.object_cols].fillna("NA")
            self.df_test[self.object_cols] = self.df_test[self.object_cols].fillna("NA")
            self._set_fix_applied("str_fillna")

    def _fix_numeric_fillna(self) -> None:
        if not self.fixes_state["numeric_fillna"]:
            self.df_baseline.fillna(0)
            self.df_test.fillna(0)
            self._set_fix_applied("numeric_fillna")

    def _fix_float_precision(self) -> None:
        if not self.fixes_state["float_precision"]:
            for col, precisions in self.float_precision_differences.items():
                min_precision = min(precisions)
                self.df_baseline[col] = self.df_baseline[col].round(min_precision)
                self.df_test[col] = self.df_test[col].round(min_precision)
            self._set_fix_applied("float_precision")

    def _fix_float_to_int(self) -> None:
        if not self.fixes_state["float_to_int"]:
            for col in self.dtype_mismatches.keys():
                if is_integer_dtype(self.df_baseline[col]) and is_float_dtype(self.df_test[col]):
                    self.df_test[col] = self.df_test[col].astype(int)
                elif is_float_dtype(self.df_baseline[col]) and is_integer_dtype(self.df_test[col]):
                    self.df_baseline[col] = self.df_baseline[col].astype(int)
            self._set_fix_applied("float_to_int")

    def train_test_split(self, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits baseline and test datasets, labels them, and returns concatenated train/test sets.

        Args:
            **kwargs: Passed to sklearn's train_test_split.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Combined train and test DataFrames.
        """
        self.fix_datasets()

        baseline_train, baseline_test = train_test_split(self.df_baseline, **kwargs)
        baseline_train["target"] = 0
        baseline_test["target"] = 0

        current_train, current_test = train_test_split(self.df_test, **kwargs)
        current_train["target"] = 1
        current_test["target"] = 1

        train = pd.concat([baseline_train, current_train]).reset_index(drop=True)
        test = pd.concat([baseline_test, current_test]).reset_index(drop=True)

        return train, test
