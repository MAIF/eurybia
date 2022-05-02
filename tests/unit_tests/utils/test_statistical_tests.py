"""
Unit test statistical_tests.py
"""
import unittest

import numpy as np

from eurybia.utils.statistical_tests import chisq_test, ksmirnov_test, compute_js_divergence, prob_mass_fun


class TestStatistical_tests(unittest.TestCase):
    """
    Unit test statistical_tests.py
    """

    def test_ksmirnov_test(self):
        """
        Unit test ksmirnov_test function
        """
        X1 = np.array([0, 3, 9857, 3444, 99, 8, 9, 9, 0, 4535, 34])
        X2 = np.array([0.45, 3, 9857, 3444, 4535, 34])

        res = ksmirnov_test(X1, X2)

        assert res["testname"] == "K-Smirnov"
        assert round(res["statistic"], 2) == 0.23
        assert round(res["pvalue"], 2) == 0.95

    def test_chisq_test(self):
        """
        Unit test chisq_test function
        """
        X1 = np.array([0, 0, 3, 4, 2, 2, 2, 2, 2, 1])
        X2 = np.array([3, 4, 3, 2, 1])

        res = chisq_test(X1, X2)

        assert res["testname"] == "Chi-Square"
        assert round(res["statistic"], 2) == 3.75
        assert round(res["pvalue"], 2) == 0.44

    def test_prob_mass_fun(self):
        """
        Unit test prob_mass_fun
        """
        X1 = np.array([0, 0, 3, 3, 0, 2, 2, 2, 1, 1])
        X2 = np.array([3, 3, 3, 2, 1])

        a = np.concatenate((X1, X2), axis=0)
        range = (a.min(), a.max())

        res1, res2 = prob_mass_fun(X1,3,range)

        assert res1.all() == np.array([0., 1., 2., 3.]).all()
        assert res2.all() == np.array([0.3, 0.2 , 0.5]).all()

    def test_compute_js_divergence(self):
        """
        Unit test Jensen Shannon divergence
        """
        X1 = np.array([0, 3, 9857, 3444, 99, 8, 9, 9, 0, 4535, 34])
        X2 = np.array([0.45, 3, 9857, 3444, 4535, 34])

        X3 = np.array([0, 0, 3, 4, 2, 2, 2, 2, 2, 1])
        X4 = np.array([3, 4, 3, 2, 1])

        res1 = compute_js_divergence(X1, X2)
        res2 = compute_js_divergence(X3, X4)

        assert round(res1, 2) == 0.17
        assert round(res2, 2) == 0.41
