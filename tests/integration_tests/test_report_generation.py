"""
Integration test for generation.py
"""

import os
import unittest
from os import path
from pathlib import Path

import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from eurybia import SmartDrift
from eurybia.report.generation import execute_report

current_path = os.path.dirname(os.path.abspath(__file__))


class TestGeneration(unittest.TestCase):
    """
    Unit test generation.py
    """

    def setUp(self):
        """
        Initialize data for testing part
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent
        titanic_original = path.join(script_path, "eurybia/data/titanicdata.csv")
        titan_df = pd.read_csv(titanic_original, index_col=0)
        y = titan_df["Survived"]
        X = titan_df.drop("Survived", axis=1).drop("Name", axis=1)
        varcat = ["Pclass", "Sex", "Embarked", "Title"]
        categ_encoding = OrdinalEncoder(cols=varcat, handle_unknown="ignore", return_df=True).fit(X)
        X_encoded = categ_encoding.transform(X)

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.75, random_state=1)
        Xtrain_e, Xtest_e, ytrain_e, ytest_e = train_test_split(X_encoded, y, train_size=0.75, random_state=1)

        rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3)
        rf.fit(Xtrain_e, ytrain_e)

        smartdrift = SmartDrift(Xtrain, Xtest, deployed_model=rf, encoding=categ_encoding)
        smartdrift.compile(full_validation=True)
        self.smartdrift = smartdrift
        self.xpl = smartdrift.xpl
        self.script_path = script_path

    def tearDown(self):
        """
        method that tidies up after the test method has been run
        """
        os.remove("./report.html")

    def test_execute_report_1(self):
        """
        Test execute_report() method
        """

        execute_report(
            smartdrift=self.smartdrift,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/project_info.yml"),
            output_file="./report.html",
        )

        assert os.path.exists("./report.html")

    def test_execute_report_2(self):
        """
        Test execute_report() method
        """

        execute_report(
            smartdrift=self.smartdrift,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/project_info.yml"),
            output_file="./report.html",
        )

        assert os.path.exists("./report.html")

    def test_execute_report_3(self):
        """
        Test execute_report() method
        """
        execute_report(
            smartdrift=self.smartdrift,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/project_info.yml"),
            output_file="./report.html",
            config=dict(title_story="Test integration", title_description="Title of test integration"),
        )

        assert os.path.exists("./report.html")

    def test_execute_report_modeldrift_1(self):
        """
        Test execute_report() method
        """
        df_perf = pd.DataFrame({"mois": [1, 2, 3], "annee": [2018, 2019, 2020], "performance": [2, 3.46, 2.5]})
        self.smartdrift.add_data_modeldrift(dataset=df_perf, metric="performance")
        execute_report(
            smartdrift=self.smartdrift,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/project_info.yml"),
            output_file="./report.html",
            config=dict(title_story="Test integration", title_description="Title of test integration"),
        )

        assert os.path.exists("./report.html")

    def test_execute_report_modeldrift_2(self):
        """
        Test execute_report() method
        """
        df_perf2 = pd.DataFrame(
            {
                "mois": [1, 2, 3],
                "annee": [2018, 2019, 2020],
                "performance": [2, 3.46, 2.5],
                "reference_column1": [1, 2, 3],
                "reference_column2": ["2018", "2019", "2020"],
            }
        )
        self.smartdrift.add_data_modeldrift(
            dataset=df_perf2, metric="performance", reference_columns=["reference_column1", "reference_column2"]
        )
        execute_report(
            smartdrift=self.smartdrift,
            explainer=self.xpl,
            project_info_file=os.path.join(current_path, "../data/project_info.yml"),
            output_file="./report.html",
            config=dict(title_story="Test integration", title_description="Title of test integration"),
        )

        assert os.path.exists("./report.html")
