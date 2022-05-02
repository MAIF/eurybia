"""
Integration test for main object Smartdrift
"""

import os
import unittest
from os import path
from pathlib import Path

import pandas as pd
import plotly
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from eurybia import SmartDrift

current_path = os.path.dirname(os.path.abspath(__file__))
TITANIC_PATH = "eurybia/data/titanicdata.csv"
TITANIC_ORIGINAL_PATH = "eurybia/data/titanic_original.csv"
TITANIC_ALTERED_PATH = "eurybia/data/titanic_altered.csv"
PROJECT_INFO_PATH = "eurybia/data/project_info_titanic.yml"
REPORT_PATH = "./test_report.html"


class TestIntegrationSmartdrift(unittest.TestCase):
    """
    Integration test smartdrift
    """

    def setUp(self):
        """
        Initialize data for testing part
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent
        titanic_csv = path.join(script_path, TITANIC_PATH)
        titanic_df = pd.read_csv(titanic_csv)
        del titanic_df["Name"]
        titanic_df = titanic_df.drop("PassengerId", axis=1)
        titanic_df_1, titanic_df_2 = train_test_split(titanic_df, test_size=0.5, random_state=42)
        titanic_df_1 = titanic_df_1.drop("Survived", axis=1)
        titanic_df_2 = titanic_df_2.drop("Survived", axis=1)
        varcat = ["Pclass", "Sex", "Embarked", "Title"]
        y = titanic_df["Survived"]
        X = titanic_df.drop("Survived", axis=1)
        categ_encoding = OrdinalEncoder(cols=varcat, handle_unknown="ignore", return_df=True).fit(X)
        X = categ_encoding.transform(X)
        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1)
        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3)
        rf.fit(x_train, y_train)
        self.titanic_df_1 = titanic_df_1
        self.titanic_df_2 = titanic_df_2
        self.rf = rf
        self.categ_encoding = categ_encoding
        self.script_path = script_path
        self.X = X

    def test_compile_smartdrift_1(self):
        """
        Test compile smartdrift
        """

        smartdrift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smartdrift.compile(full_validation=False)
        fig_continuous = smartdrift.plot.generate_fig_univariate("Age")
        fig_categorical = smartdrift.plot.generate_fig_univariate("Pclass")

        assert smartdrift.auc < 0.6

        assert isinstance(fig_continuous, plotly.graph_objs._figure.Figure)
        assert isinstance(fig_categorical, plotly.graph_objs._figure.Figure)

    def test_compile_smartdrift_2(self):
        """
        Test compile smartdrift
        """

        smartdrift = SmartDrift(
            self.titanic_df_1, self.titanic_df_2, deployed_model=self.rf, encoding=self.categ_encoding
        )
        smartdrift.compile(full_validation=True)
        fig_continuous = smartdrift.plot.generate_fig_univariate("Age")
        fig_categorical = smartdrift.plot.generate_fig_univariate("Pclass")
        fig_predict = smartdrift.plot.generate_fig_univariate(df_all=smartdrift.df_predict, col="Score", hue="dataset")

        assert smartdrift.auc < 0.6
        assert (
            smartdrift.feature_importance.loc[smartdrift.feature_importance["feature"] == "Age", "deployed_model"].iloc[
                0
            ]
            > 0.1
        )
        assert (
            smartdrift.feature_importance.loc[
                smartdrift.feature_importance["feature"] == "Age", "datadrift_classifier"
            ].iloc[0]
            > 0.1
        )

        assert isinstance(fig_continuous, plotly.graph_objs._figure.Figure)
        assert isinstance(fig_categorical, plotly.graph_objs._figure.Figure)
        assert isinstance(fig_predict, plotly.graph_objs._figure.Figure)

    def test_compile_smartdrift_3(self):
        """
        Test compile smartdrift
        """

        smartdrift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smartdrift.compile(date_compile_auc="01/01/2021", datadrift_file="datadrift_metric.csv")

        smartdrift2 = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smartdrift2.compile(date_compile_auc="01/01/2022", datadrift_file="datadrift_metric.csv")

        fig_continuous = smartdrift2.plot.generate_fig_univariate("Age")
        fig_categorical = smartdrift2.plot.generate_fig_univariate("Pclass")

        assert smartdrift2.auc < 0.6

        assert isinstance(fig_continuous, plotly.graph_objs._figure.Figure)
        assert isinstance(fig_categorical, plotly.graph_objs._figure.Figure)
        os.remove("datadrift_metric.csv")
