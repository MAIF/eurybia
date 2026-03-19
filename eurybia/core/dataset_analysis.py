"""dataset_analysis module"""

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_float_dtype, is_string_dtype


class DatasetAnalysis:
    """
    DatasetAnalysis provides tools for comparing and analyzing differences between two pandas DataFrames,
    typically representing a baseline and a test dataset. It supports detection of schema and data
    inconsistencies, such as column mismatches, data type differences, missing values, and float precision issues.

    Key Features:
    - Identifies intersecting, new, and removed columns between datasets.
    - Detects data type mismatches and float precision differences.
    - Analyzes categorical value differences between datasets.
    - Provides utilities for filling missing values and aligning schemas.

    Attributes:
        ignored_cols (list[str]): List of columns ignored in analysis.
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

    Methods:
        clean_datasets(): Applies configured data cleaning and transformation fixes to the datasets.
    """

    def __init__(
        self,
        df_test: pd.DataFrame,
        df_baseline: pd.DataFrame,
        ignored_cols: list[str] | None = None,
        sample_size: int | None = None,
        random_state: int = 42,
    ):
        """
        Initializes the dataset analysis object with test and baseline DataFrames,
        applies optional sampling and ignores specified columns.

        Args:
            df_test (pd.DataFrame): The test dataset to analyze.
            df_baseline (pd.DataFrame): The baseline dataset for comparison.
            ignored_cols (list[str] | None, optional): List of column names to ignore in both datasets.
            sample_size (int | None, optional): If provided, limits both datasets to this number of samples.
            random_state (int, optional): Random seed used when sampling rows from the datasets.
        """
        self._ignored_cols: list[str] = ignored_cols if ignored_cols is not None else []
        self._df_test = df_test.drop(self._ignored_cols, axis=1, errors="ignore")
        self._df_baseline = df_baseline.drop(self._ignored_cols, axis=1, errors="ignore")

        if sample_size is not None:
            if self._df_test.shape[0] > sample_size:
                self._df_test = self._df_test.sample(sample_size, random_state=random_state)
            if self._df_baseline.shape[0] > sample_size:
                self._df_baseline = self._df_baseline.sample(sample_size, random_state=random_state)

    @property
    def ignored_cols(self) -> list[str]:
        """Getter"""
        return self._ignored_cols

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
        return [c for c in self._df_test.columns if c in self._df_baseline.columns]

    @property
    def new_columns(self) -> list[str]:
        """Identifies columns that have been added by comparing df_test and df_baseline.

        Returns:
            list[str]: A list of column names that exist in df_test but not in df_baseline.
        """
        return [c for c in self._df_test.columns if c not in self._df_baseline.columns]

    @property
    def removed_columns(self) -> list[str]:
        """Identifies columns that have been removed by comparing df_baseline and df_test.

        Returns:
            list[str]: A list of column names that exist in df_baseline but not in df_test.
        """
        return [c for c in self._df_baseline.columns if c not in self._df_test.columns]

    def _has_same_dtype(self, colname: str) -> bool:
        """Check if the specified column has the same data type in both the baseline and test DataFrames.

        Args:
            colname (str): The name of the column to compare.
        Returns:
            bool: True if the column has the same data type in both DataFrames, False otherwise.
        """

        return self._df_baseline[colname].dtype == self._df_test[colname].dtype

    @property
    def dtype_mismatches(self) -> dict[str, tuple[str, str]]:
        """Identifies columns with mismatched data types between the baseline and test DataFrames.

        Returns:
            dict[str, tuple[str, str]]:
                A dictionary where each key is a column name with a data type mismatch,
                and the value is a tuple containing the data types from the test and baseline DataFrames, respectively.
        """
        dtype_mismatches: dict[str, tuple[str, str]] = dict()
        for colname in self.intersecting_columns:
            if not self._has_same_dtype(colname):
                dtype_mismatches[colname] = (
                    str(self._df_test[colname].dtype),
                    str(self._df_baseline[colname].dtype),
                )
        return dtype_mismatches

    @property
    def object_cols(self) -> list[str]:
        """Returns a list of column names from the intersecting and not dtype mismatching columns
        whose data type is string.

        Returns:
            list[str]: List of column names with 'string' dtype and no dtype mismatches.
        """
        return [colname for colname in self.valid_columns if is_string_dtype(self._df_baseline[colname].dtype)]

    @property
    def float_cols(self) -> list[str]:
        """Returns a list of column names from the intersecting and not dtype mismatching columns
        whose data type is float.

        Returns:
            list[str]: List of column names with 'float' dtype and no dtype mismatches.
        """

        return [colname for colname in self.valid_columns if is_float_dtype(self._df_baseline[colname])]

    @property
    def datetime_cols(self) -> list[str]:
        """Returns a list of column names from the intersecting and not dtype mismatching columns
        whose data type is datetime.

        Returns:
            list[str]: List of column names with datetime dtype and no dtype mismatches.
        """
        return [colname for colname in self.valid_columns if is_datetime64_any_dtype(self._df_baseline[colname])]

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
            baseline_modalities = pd.Series(self._df_baseline[colname]).dropna().unique()
            test_modalities = pd.Series(self._df_test[colname]).dropna().unique()

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
                  of (test dataset max precision, baseline dataset max precision) as integers.
        """

        def nb_digits(i: float):
            split_i = str(i).split(".")
            return len(split_i[1]) if len(split_i) > 1 else 0

        different_float_precision: dict[str, tuple[int, int]] = dict()

        for colname in self.float_cols:
            baseline_digits = self._df_baseline[colname].apply(nb_digits)
            test_digits = self._df_test[colname].apply(nb_digits)
            baseline_max_precision = int(np.max(baseline_digits))
            test_max_precision = int(np.max(test_digits))
            if baseline_max_precision != test_max_precision:
                different_float_precision[colname] = (test_max_precision, baseline_max_precision)

        return different_float_precision

    def clean_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Cleans and preprocesses the test and baseline datasets.

        The cleaning steps include:
            1. Keeping only columns with matching names and types in both datasets.
            2. Splitting datetime columns into separate year, month, and day columns.
            3. Filling missing values in string columns.
            4. Filling missing values in numeric columns.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                A tuple containing the cleaned test and baseline DataFrames.
        """

        df_test = self._df_test
        df_baseline = self._df_baseline

        # keep only columns of same name and type
        df_test = self._fix_valid_cols(df_test)
        df_baseline = self._fix_valid_cols(df_baseline)

        # split datetime cols in three (year, month, day) cols
        df_test = self._fix_date_split(df_test)
        df_baseline = self._fix_date_split(df_baseline)

        # fill str cols missing values
        df_test = self._fix_str_fillna(df_test)
        df_baseline = self._fix_str_fillna(df_baseline)

        # fill numeric cols missing values
        df_test = self._fix_numeric_fillna(df_test)
        df_baseline = self._fix_numeric_fillna(df_baseline)

        return df_test, df_baseline

    def _fix_str_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.object_cols] = df[self.object_cols].fillna("NA")
        return df

    def _fix_numeric_fillna(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(0, inplace=False)

    def _fix_valid_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.valid_columns]

    def _fix_date_split(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.datetime_cols:
            df[col + "_year"] = df[col].dt.year
            df[col + "_month"] = df[col].dt.month
            df[col + "_day"] = df[col].dt.day
            df = df.drop(col, axis=1)
        return df
