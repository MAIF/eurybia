"""
Unit tests for SmartPlotter.py
"""

import copy
import unittest
from os import path
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly
from sklearn.model_selection import train_test_split

from eurybia.core.smartdrift import SmartDrift
from eurybia.report.common import VarType
from eurybia.style.style_utils import colors_loading, select_palette

TITANIC_PATH = "eurybia/data/titanicdata.csv"


class TestSmartPlotter(unittest.TestCase):
    """
    Unit test SmartPlotter.py
    """

    def setUp(self):
        """SetUp"""

        script_path = Path(path.abspath(__file__)).parent.parent.parent.parent
        titanic_csv = path.join(script_path, TITANIC_PATH)
        titanic_df = pd.read_csv(titanic_csv)
        titanic_df_1, titanic_df_2 = train_test_split(titanic_df, test_size=0.5, random_state=42)
        self.smartdrift = SmartDrift(titanic_df_1, titanic_df_2)

    @patch("eurybia.core.smartplotter.SmartPlotter.generate_fig_univariate_continuous")
    @patch("eurybia.core.smartplotter.SmartPlotter.generate_fig_univariate_categorical")
    def test_generate_fig_univariate_1(self, mock_plot_cat, mock_plot_cont):
        """
        Unit test for generate_fig_univariate()'s method
        """
        df = pd.DataFrame(
            {
                "string_data": ["a", "b", "c", "d", "e", np.nan],
                "data_train_test": ["train", "train", "train", "train", "test", "test"],
            }
        )
        dict_color_palette = {
            "df_bBaseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_Current": (244 / 255, 192 / 255, 0),
            "true": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "pred": (244 / 255, 192 / 255, 0),
            "train": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "test": (244 / 255, 192 / 255, 0),
            "df_baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_current": (244 / 255, 192 / 255, 0),
        }

        self.smartdrift.plot.generate_fig_univariate(
            "string_data", "data_train_test", df, dict_color_palette=dict_color_palette
        )
        mock_plot_cat.assert_called_once()
        self.assertEqual(mock_plot_cont.call_count, 0)

    @patch("eurybia.core.smartplotter.SmartPlotter.generate_fig_univariate_continuous")
    @patch("eurybia.core.smartplotter.SmartPlotter.generate_fig_univariate_categorical")
    def test_generate_fig_univariate_2(self, mock_plot_cat, mock_plot_cont):
        """
        Unit test for generate_fig_univariate()'s method
        """
        df = pd.DataFrame(
            {"int_data": list(range(50)), "data_train_test": ["train", "train", "train", "train", "test"] * 10}
        )

        dict_color_palette = {
            "df_Baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_Current": (244 / 255, 192 / 255, 0),
            "true": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "pred": (244 / 255, 192 / 255, 0),
            "train": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "test": (244 / 255, 192 / 255, 0),
            "df_baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_current": (244 / 255, 192 / 255, 0),
        }

        self.smartdrift.plot.generate_fig_univariate(
            "int_data", "data_train_test", df, dict_color_palette=dict_color_palette
        )
        mock_plot_cont.assert_called_once()
        self.assertEqual(mock_plot_cat.call_count, 0)

    @patch("eurybia.core.smartplotter.SmartPlotter.generate_fig_univariate_continuous")
    @patch("eurybia.core.smartplotter.SmartPlotter.generate_fig_univariate_categorical")
    def test_generate_fig_univariate_3(self, mock_plot_cat, mock_plot_cont):
        """
        Unit test for generate_fig_univariate()'s method
        """
        df = pd.DataFrame(
            {
                "int_cat_data": [10, 10, 20, 20, 20, 10],
                "data_train_test": ["train", "train", "train", "train", "test", "test"],
            }
        )
        dict_color_palette = {
            "df_Baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_Current": (244 / 255, 192 / 255, 0),
            "true": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "pred": (244 / 255, 192 / 255, 0),
            "train": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "test": (244 / 255, 192 / 255, 0),
            "df_baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_current": (244 / 255, 192 / 255, 0),
        }

        self.smartdrift.plot.generate_fig_univariate(
            "int_cat_data", "data_train_test", df, dict_color_palette=dict_color_palette
        )
        mock_plot_cat.assert_called_once()
        self.assertEqual(mock_plot_cont.call_count, 0)

    def test_generate_fig_univariate_continuous(self):
        """
        Unit test for generate_fig_univariate_continuous()'s method
        """
        df = pd.DataFrame(
            {
                "int_data": [10, 20, 30, 40, 50, 0],
                "data_train_test": ["train", "train", "train", "train", "test", "test"],
            }
        )
        dict_color_palette = {
            "df_Baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_Current": (244 / 255, 192 / 255, 0),
            "true": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "pred": (244 / 255, 192 / 255, 0),
            "train": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "test": (244 / 255, 192 / 255, 0),
            "df_baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_current": (244 / 255, 192 / 255, 0),
        }
        fig = self.smartdrift.plot.generate_fig_univariate_continuous(
            df, "int_data", "data_train_test", dict_color_palette=dict_color_palette
        )
        assert isinstance(fig, plotly.graph_objs._figure.Figure)

    def test_generate_fig_univariate_categorical(self):
        """
        Unit test for generate_fig_univariate_categorical()'s method
        """
        df = pd.DataFrame(
            {"int_data": [0, 0, 0, 1, 1, 0], "data_train_test": ["train", "train", "train", "train", "test", "test"]}
        )
        dict_color_palette = {
            "df_Baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_Current": (244 / 255, 192 / 255, 0),
            "true": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "pred": (244 / 255, 192 / 255, 0),
            "train": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "test": (244 / 255, 192 / 255, 0),
            "df_baseline": (74 / 255, 99 / 255, 138 / 255, 0.7),
            "df_current": (244 / 255, 192 / 255, 0),
        }

        fig = self.smartdrift.plot.generate_fig_univariate_categorical(
            df, "int_data", "data_train_test", dict_color_palette=dict_color_palette
        )
        nb_bars = len(fig.to_dict()["data"][0]["y"]) + len(fig.to_dict()["data"][1]["y"])
        assert nb_bars == 4  # Number of bars

    def test_scatter_feature_importance(self):
        """
        Unit test for scatter_feature_importance()'s method
        """
        importances = {
            "feature": ["Parch", "Embarked", "SibSp", "Age", "Fare", "Pclass", "Title", "Sex"],
            "deployed_model": [0.014573, 0.046866, 0.054405, 0.068945, 0.102350, 0.155283, 0.255974, 0.263060],
            "datadrift_classifier": [0.045921, 0.118159, 0.018614, 0.137789, 0.184038, 0.143481, 0.000000, 0.162925],
        }
        features_importances = pd.DataFrame(importances)
        stat_test = {
            "feature": ["Parch", "Embarked", "SibSp", "Age", "Fare", "Pclass", "Title", "Sex"],
            "testname": [
                "Chi-Square",
                "Chi-Square",
                "K-Smirnov",
                "Chi-Square",
                "Chi-Square",
                "K-Smirnov",
                "Chi-Square",
                "Chi-Square",
            ],
            "statistic": [
                2.6092852997885805,
                1.0551275457681237,
                0.05108076787423792,
                11.737348438770884,
                3.7354255436570627,
                0.05752002821585126,
                3.0136100828147927,
                13.864955367956966,
            ],
            "pvalue": [
                0.2712694558959706,
                0.30432911545137264,
                0.5753367740062436,
                0.06809137774132133,
                0.7124287331705583,
                0.4263491432475492,
                0.22161690489144445,
                0.6087737509479118,
            ],
        }
        statistical_tests = pd.DataFrame(stat_test)

        fig = self.smartdrift.plot.scatter_feature_importance(
            feature_importance=features_importances.set_index("feature"),
            datadrift_stat_test=statistical_tests.set_index("feature"),
        )
        assert isinstance(fig, plotly.graph_objs._figure.Figure)
        assert len(fig.data) == 1
        assert len(fig.data[0]["x"]) == features_importances.shape[0]

    def test_generate_historical_datadrift_metric_1(self):
        """
        Unit test for generate_historical_datadrift_metric()'s method
        """
        auc = {
            "date": ["23/09/2021", "23/08/2021", "23/07/2021", "23/06/2021"],
            "auc": [0.528107, 0.532106, 0.510653, 0.48],
        }
        datadrift_historical = pd.DataFrame(auc)
        fig = self.smartdrift.plot.generate_historical_datadrift_metric(datadrift_historical)
        assert isinstance(fig, plotly.graph_objs._figure.Figure)
        np.testing.assert_allclose(fig.data[0].y, [0.48, 0.510653, 0.532106, 0.528107])
        np.testing.assert_allclose(fig.data[0].text, [0.48, 0.51, 0.53, 0.53])
        pd.testing.assert_frame_equal(datadrift_historical, pd.DataFrame(auc))

    def test_generate_historical_datadrift_metric_2(self):
        """
        Unit test for generate_historical_datadrift_metric()'s method
        """
        auc = {
            "date": ["23/09/2021", "23/08/2021", "23/07/2021", "23/06/2021"],
            "auc": [0.528107, 0.532106, 0.510653, 0.48],
            "JS_predict": [0.28107, 0.32106, 0.50653, 0.2],
        }
        datadrift_historical = pd.DataFrame(auc)
        self.smartdrift.deployed_model = object()
        fig = self.smartdrift.plot.generate_historical_datadrift_metric(datadrift_historical)
        assert isinstance(fig, plotly.graph_objs._figure.Figure)
        np.testing.assert_allclose(fig.data[0].y, [0.48, 0.510653, 0.532106, 0.528107])
        np.testing.assert_allclose(fig.data[1].y, [0.2, 0.50653, 0.32106, 0.28107])
        assert fig.layout.yaxis2.range == (0, 1)
        pd.testing.assert_frame_equal(datadrift_historical, pd.DataFrame(auc))

    def test_generate_modeldrift_data_1(self):
        """
        Unit test for generate_modeldrift_data()'s method
        """
        lift_info = {
            "Date": ["01/07/2019", "01/08/2019"],
            "plage_historique": [1, 1],
            "target": ["souscription", "souscription"],
            "type_lift": ["lift10", "lift10"],
            "lift": [3.32, 3.18],
        }
        df_lift = pd.DataFrame(lift_info)

        fig = self.smartdrift.plot.generate_modeldrift_data(df_lift, metric="lift")
        assert isinstance(fig, plotly.graph_objs._figure.Figure)

    def test_generate_modeldrift_data_2(self):
        """
        Unit test for generate_modeldrift_data()'s method
        """
        lift_info = {
            "Date": ["01/07/2019", "01/08/2019"],
            "plage_historique": [1, 1],
            "target": ["souscription", "souscription"],
            "type_lift": ["lift10", "lift10"],
            "lift": [3.32, 3.18],
        }
        df_lift = pd.DataFrame(lift_info)

        fig = self.smartdrift.plot.generate_modeldrift_data(
            df_lift, metric="lift", reference_columns=["plage_historique", "target", "type_lift"]
        )
        assert isinstance(fig, plotly.graph_objs._figure.Figure)

    def test_define_style_attributes(self):
        # clear style attributes
        self.smartdrift.plot.style_dict = dict()

        colors_dict = copy.deepcopy(select_palette(colors_loading(), "eurybia"))
        self.smartdrift.plot.define_style_attributes(colors_dict=colors_dict)

        assert hasattr(self.smartdrift.plot, "_style_dict")
        assert len(list(self.smartdrift.plot.style_dict.keys())) > 0

    def test_generate_indicator(self):
        """
        Unit test for generate_indicator()'s method
        """
        self.smartdrift.compile()
        svg = self.smartdrift.plot.generate_indicator(fig_value=self.smartdrift.auc, height=300, width=500, title="AUC")
        assert isinstance(svg, str)
        assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
        assert "AUC" in svg
        assert svg.count('class="gauge-tick"') == 11
        assert svg.count('class="gauge-tick-label"') == 11
        assert 'class="gauge-tick-label" x="250.00"' in svg
        assert 'font-size="15" fill="#555">' in svg
        assert ">0.75</text>" in svg
        assert 'stroke-width="52.00"' in svg
        assert 'class="gauge-fill"' in svg
        assert 'stroke="#111" stroke-width="28.60" stroke-linecap="butt"' in svg
        assert "M 106.00 228.00 A 144.00 144.00 0 0 1" in svg
        assert 'class="gauge-needle"' not in svg
        assert 'class="gauge-value" x="250"' in svg
        assert 'class="gauge-value" x="250" y="200.80"' in svg
        assert 'x="106" y="251" text-anchor="middle"' in svg
        assert 'x="394" y="251" text-anchor="middle"' in svg
        assert svg.count('class="gauge-track"') == 1
        assert "A 144.00 144.00 0 0 1 394.00 228.00" in svg
        assert "gauge-segment" not in svg
        assert 'font-size="56" font-weight="600"' in svg
