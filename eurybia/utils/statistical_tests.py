"""
Statistical test functions
"""

import numpy as np
import pandas as pd
import scipy.spatial
from scipy import stats


def ksmirnov_test(obs_a: np.array, obs_b: np.array) -> dict:
    """
    Returns a dict containing testname, statistic, pvalue of the ks test

    Parameters
    ----------
    obs_a : np.array
        1D array containing the feature values in the first sample
    obs_b : np.array
        1D array containing the feature values \u200b\u200bin the second sample

    Returns
    -------
    dict :
        3 keys : testname, statistic, pvalue
    """
    test_result = stats.ks_2samp(obs_a, obs_b)
    output = {"testname": "K-Smirnov", "statistic": test_result.statistic, "pvalue": test_result.pvalue}
    return output


def chisq_test(obs_a: np.array, obs_b: np.array) -> dict:
    """
    Returns a dict containing testname, statistic, pvalue of the chisquare test

    Parameters
    ----------
    obs_a : np.array
        1D array containing the feature values in the first sample
    obs_b : np.array
        1D array containing the feature values \u200b\u200bin the second sample

    Returns
    -------
    dict :
        3 keys : testname, statistic, pvalue
    """
    uniq_a, freq_a = np.unique(obs_a, return_counts=True)
    freq_a_df = pd.DataFrame(freq_a, index=uniq_a, columns=["a"])

    uniq_b, freq_b = np.unique(obs_b, return_counts=True)
    freq_b_df = pd.DataFrame(freq_b, index=uniq_b, columns=["b"])

    freq = pd.concat([freq_a_df, freq_b_df], axis=1).transpose().to_numpy(na_value=0)

    g, p, _, _ = stats.chi2_contingency(freq)

    output = {"testname": "Chi-Square", "statistic": g, "pvalue": p}
    return output


def prob_mass_fun(data, n, range):
    """
    Computing the probability mass function using NumPy’s histogram.

    Parameters
    ----------
    data: pandas.Series
        Series to split
    n: int
        The number of equal-width bins in the given range.
    range: tuple
        The lower and upper range of the bins.
    Returns
    -------
    e, Return the bin edges (length= n+1).
    p, Return the frequencies of each interval (length= n).

    """
    h, e = np.histogram(data, n, range)
    p = h / data.shape[0]
    return e, p


def compute_js_divergence(df_1, df_2, n_bins=30):
    """
    Computing the Jensen-Shannon divergence between 2 dataframe

    Parameters
    ----------
    df_1: pandas.Series
        Series to compare
    df_2: pandas.DataFrame
        Series to compare
    n_bins: int
        The number of equal-width bins in the given range
    Returns
    -------
    jensenshannon(p, q) : score between 0 (The two score distributions are identical)
    and 1 (The two score distributions maximally different).
    """
    a = np.concatenate((df_1, df_2), axis=0)
    e, p = prob_mass_fun(df_1, n=n_bins, range=(a.min(), a.max()))
    _, q = prob_mass_fun(df_2, n=e, range=(a.min(), a.max()))

    return scipy.spatial.distance.jensenshannon(p, q)
