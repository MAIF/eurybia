"""
Utils is a group of function for the library
"""
from pathlib import Path

import pandas as pd


def convert_string_to_int_keys(input_dict: dict) -> dict:
    """
    Returns the dict with integer keys instead of string keys
    Parameters
    ----------
    input_dict: dict
    Returns
    -------
    dict
    """
    return {int(k): v for k, v in input_dict.items()}


def base_100(series: pd.Series) -> pd.Series:
    """
    base_100 function put a pd.Series in base 100
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


def get_project_root():
    """
    Returns project root absolute path.
    """
    current_path = Path(__file__)
    return current_path.parent.parent.resolve()


def truncate_str(text, maxlen=40):
    """
    truncate a string
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


def round_to_k(x, k):
    """
    round float to k significant figure
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
    x = float(x)
    new_x = float("%s" % float(f"%.{k}g" % x))  # Rounding to k important figures
    if new_x % 1 == 0:
        return int(new_x)  # Avoid the '.0' that can mislead the user that it may be a round number
    else:
        return new_x
