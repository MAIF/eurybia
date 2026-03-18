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

    # def convert_date_col_into_multiple_col(df: pd.DataFrame) -> pd.DataFrame:
    #     """Transform datetime column into multiple columns
    #         - year
    #         - month
    #         - day
    #     Drop datetime column

    #     Parameters
    #     ----------
    #     df: pd.Dataframe
    #        input DataFrame with datetime columns

    #     Returns
    #     -------
    #     pd.Dataframe
    #         DataFrame without datetime columns

    #     """
    #     date_col_list = [column for column in df.columns if is_datetime(df[column])]

    #     for col_date in date_col_list:
    #         df[col_date + "_year"] = df[col_date].dt.year
    #         df[col_date + "_month"] = df[col_date].dt.month
    #         df[col_date + "_day"] = df[col_date].dt.day

    #         # droping original date column
    #         df = df.drop(col_date, axis=1)

    #     return df


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


def train_test_split_concat(df_baseline, df_test, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the baseline and test datasets into train and test sets, labels them,
    and returns the concatenated train and test DataFrames.

    Parameters:
        **kwargs: Additional keyword arguments passed to sklearn's train_test_split (e.g., test_size, random_state).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
        - train: Concatenated DataFrame of training samples from both baseline and test datasets,
            with a 'target' column indicating source (0 for baseline, 1 for test).
        - test: Concatenated DataFrame of test samples from both baseline and test datasets,
            with a 'target' column indicating source (0 for baseline, 1 for test).
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
