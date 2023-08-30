"""
Unit tests for project_report.py
"""
import os
import unittest
from os import path
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import plotly
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from eurybia import SmartDrift
from eurybia.report.project_report import DriftReport

expected_attrs = ["smartdrift", "explainer", "config_report"]

current_path = os.path.dirname(os.path.abspath(__file__))


class TestDriftReport(unittest.TestCase):
    """
    Unit tests for project_report.py
    """

    def setUp(self):
        """
        Use a setUp to init attribute to use in unit test
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent.parent
        titanic_original = path.join(script_path, "eurybia/data/titanicdata.csv")
        titan_df = pd.read_csv(titanic_original, index_col=0)
        y = titan_df["Survived"]
        X = titan_df.drop("Survived", axis=1).drop("Name", axis=1)
        varcat = ["Pclass", "Sex", "Embarked", "Title"]
        categ_encoding = OrdinalEncoder(cols=varcat, handle_unknown="ignore", return_df=True).fit(X)
        x_encoded = categ_encoding.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1)
        x_train_e, x_test_e, y_train_e, y_test_e = train_test_split(x_encoded, y, train_size=0.75, random_state=1)

        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3)
        rf.fit(x_train_e, y_train_e)

        smartdrift = SmartDrift(x_train, x_test, deployed_model=rf, encoding=categ_encoding)
        smartdrift.compile(full_validation=True)

        lift_info = {
            "annee": ["2019", "2019"],
            "mois": ["7", "8"],
            "plage_historique": [1, 1],
            "target": ["souscription", "souscription"],
            "lift": [3.32, 3.18],
        }
        df_lift = pd.DataFrame(lift_info)
        smartdrift.add_data_modeldrift(dataset=df_lift, metric="lift", reference_columns=["target", "plage_historique"])
        self.project_info_file = os.path.join(script_path, "eurybia/data/project_info_titanic.yml")
        self.x_train = x_train
        self.x_test = x_test
        self.rf = rf
        self.categ_encoding = categ_encoding
        self.df_list = df_lift
        self.report = DriftReport(
            smartdrift=smartdrift,
            explainer=smartdrift.xpl,
            project_info_file=self.project_info_file,
            config_report=dict(title_story="Drift Report", title_description="Test drift report"),
        )

    def test_init_1(self):
        """
        Test init() of DriftReport Class
        """
        report = self.report
        for attr in expected_attrs:
            assert hasattr(report, attr)

    def test_init_2(self):
        """
        Test init() of DriftReport Class
        """
        assert isinstance(self.report.features_imp_list, list)

    def test_init_4(self):
        """
        Test init() of DriftReport Class
        """
        assert self.report.title_story == "Drift Report"

    def test_create_data_drift_1(self):
        """
        test _create_data_drift() method
        """
        report = self.report
        df_baseline = pd.DataFrame({"data_drift_split": ["test", "test"]})
        df_current = pd.DataFrame({"data_drift_split": ["test", "test"]})
        dataset_names = pd.DataFrame({"df_current": "Current dataset", "df_baseline": "Historical dataset"}, index=[0])

        with self.assertRaises(ValueError):
            report._create_data_drift(df_current, df_baseline, dataset_names)

    def test_create_data_drift_2(self):
        """
        test _create_data_drift() method
        """
        report = self.report
        df_baseline = None
        df_current = None
        dataset_names = None
        assert report._create_data_drift(df_current, df_baseline, dataset_names) is None

    def test_display_model_contribution(self):
        """
        Test display_model_contribution() method
        """
        self.report.display_model_contribution()

    def test_display_data_modeldrift_1(self):
        """
        Test display_data_modeldrift() method
        """
        fig = self.report.display_data_modeldrift()
        assert isinstance(fig[0][0], plotly.graph_objs._figure.Figure)

    def test_display_data_modeldrift_2(self):
        """
        Test display_data_modeldrift() method
        """
        smartdrift = SmartDrift(self.x_train, self.x_test, deployed_model=self.rf, encoding=self.categ_encoding)
        smartdrift.compile()
        lift_info = {
            "annee": ["2019", "2019"],
            "mois": ["7", "8"],
            "plage_historique": [1, 1],
            "target": ["souscription", "souscription"],
            "lift": [3.32, 3.18],
        }
        df_lift = pd.DataFrame(lift_info)
        smartdrift.add_data_modeldrift(dataset=df_lift, metric="lift")
        report = DriftReport(
            smartdrift=smartdrift,
            explainer=smartdrift.xpl,
            config_report=dict(title_story="Drift Report", title_description="Test drift report"),
        )
        fig = report.display_data_modeldrift()
        assert isinstance(fig[0][0], plotly.graph_objs._figure.Figure)
