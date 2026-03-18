"""Utils is a group of function for the library"""

from math import floor, log10
from pathlib import Path

import pandas as pd
from pandas.api.types import is_object_dtype
from sklearn.model_selection import train_test_split


def convert_string_to_int_keys(input_dict: dict) -> dict:
    """Returns the dict with integer keys instead of string keys

    Parameters
    ----------
    input_dict: dict

    Returns
    -------
    dict

    """
    return {int(k): v for k, v in input_dict.items()}


def base_100(series: pd.Series) -> pd.Series:
    """base_100 function put a pd.Series in base 100

    Parameters
    ----------
    serie: pd.Series
       input series to convert to base 100

    Returns
    -------
    pd.Series
        converted series

    """
    tot = series.sum()
    return series / tot


def get_project_root() -> Path:
    """Returns project root absolute path."""
    current_path = Path(__file__)
    return current_path.parent.parent.resolve()


def truncate_str(text: str, maxlen: int = 40) -> str:
    """Truncate a string

    Parameters
    ----------
    text : string
        string to check in order to add line break
    maxlen : int
        number of characters before truncation

    Returns
    -------
    string
        truncated text

    """
    if isinstance(text, str) and len(text) > maxlen:
        tot_length = 0
        input_words = text.split()
        output_words = []
        for word in input_words[:-1]:
            tot_length = tot_length + len(word)
            if tot_length <= maxlen:
                output_words.append(word)

        text = " ".join(output_words)
        if len(input_words) > len(output_words):
            text = text + "..."
    return text


def round_to_k(x: float, k: int) -> float | int:
    """Round float to k significant figure

    Parameters
    ----------
    x : float
        number to round
    k : int
        the number of significant figures

    Returns
    -------
    float or int

    """
    if x == 0:
        return 0
    new_x = round(x, k - int(floor(log10(abs(x)))) - 1)

    if new_x % 1 == 0:
        return int(new_x)  # Avoid the '.0' that can mislead the user that it may be a round number
    else:
        return new_x


def cat_features_indices(df: pd.DataFrame) -> list[int]:
    """Returns the indices of categorical features in a pandas DataFrame.

    A categorical feature is identified as a column with an object dtype.
    Parameters:
        df (pd.DataFrame): The input DataFrame to analyze.
    Returns:
        list[int]: A list of indices corresponding to categorical columns in the DataFrame.
    """

    i = 0
    indice_cat = []
    for col in df.columns:
        if col in [colname for colname in df.columns if is_object_dtype(df[colname].dtype)]:
            indice_cat.append(i)
        i += 1
    return indice_cat


def train_test_split_concat(
    df_baseline: pd.DataFrame, df_test: pd.DataFrame, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits two DataFrames (baseline and test) into train and test sets, assigns a target label to each,
    and concatenates the corresponding splits.

    The function performs a train-test split on both `df_baseline` and `df_test` using the provided
    keyword arguments (passed to `sklearn.model_selection.train_test_split`). It then assigns a
    target column with value 0 to the baseline data and 1 to the test data. The resulting train and
    test sets are concatenated and returned.

    Args:
        df_baseline (pd.DataFrame): The baseline DataFrame to split and label as target 0.
        df_test (pd.DataFrame): The test DataFrame to split and label as target 1.
        **kwargs: Additional keyword arguments passed to `train_test_split`.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the concatenated train and test DataFrames,
        each with a 'target' column indicating the source (0 for baseline, 1 for test).
    """
    baseline_train, baseline_test = train_test_split(df_baseline, **kwargs)
    baseline_train["target"] = 0
    baseline_test["target"] = 0

    current_train, current_test = train_test_split(df_test, **kwargs)
    current_train["target"] = 1
    current_test["target"] = 1

    train = pd.concat([baseline_train, current_train]).reset_index(drop=True)
    test = pd.concat([baseline_test, current_test]).reset_index(drop=True)

    return train, test
