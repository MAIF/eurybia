"""
Unit tests for SmartDrift
"""
import os
import unittest
from os import path
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import shapash
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from eurybia import SmartDrift
from eurybia.style.style_utils import colors_loading, select_palette

TITANIC_PATH = "eurybia/data/titanicdata.csv"
TITANIC_ORIGINAL_PATH = "eurybia/data/titanic_original.csv"
TITANIC_ALTERED_PATH = "eurybia/data/titanic_altered.csv"
PROJECT_INFO_PATH = "eurybia/data/project_info_titanic.yml"
REPORT_PATH = "./test_report.html"


class TestSmartDrift(unittest.TestCase):
    """
    Unit test on SmartDrift class
    """

    def setUp(self):
        """SetUp"""
        script_path = Path(path.abspath(__file__)).parent.parent.parent.parent
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

    def test_init_1(self):
        """
        test init 1 SmartDrift
        """
        smart_drift = SmartDrift()
        assert hasattr(smart_drift, "df_baseline")

    def test_compile_nooptions(self):
        """
        Test compile()
        """
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smart_drift.compile()
        assert isinstance(smart_drift.xpl, shapash.explainer.smart_explainer.SmartExplainer)

    def test_compile_fullvalid(self):
        """
        test compile() with full validation
        """
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smart_drift.compile(full_validation=True)
        assert isinstance(smart_drift.xpl, shapash.explainer.smart_explainer.SmartExplainer)

    def test_compile_nosampling(self):
        """
        test compile() without no sampling
        """
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smart_drift.compile(full_validation=True, sampling=False)
        assert isinstance(smart_drift.xpl, shapash.explainer.smart_explainer.SmartExplainer)

    def test_compile_samplingsize(self):
        """
        test compile() with a fixed sample size
        """
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smart_drift.compile(full_validation=True, sample_size=100)
        assert isinstance(smart_drift.xpl, shapash.explainer.smart_explainer.SmartExplainer)

    def test_compile_model_encoder(self):
        """
        test compile() with a model and an encoder specified
        """
        smart_drift = SmartDrift(
            self.titanic_df_1, self.titanic_df_2, deployed_model=self.rf, encoding=self.categ_encoding
        )
        smart_drift.compile()
        assert isinstance(smart_drift.xpl, shapash.explainer.smart_explainer.SmartExplainer)

    def test_compile_dataset_names(self):
        """
        test compile() with a model and an encoder specified
        """
        smart_drift = SmartDrift(
            self.titanic_df_1,
            self.titanic_df_2,
            deployed_model=self.rf,
            encoding=self.categ_encoding,
            dataset_names={"df_current": "titanic 2", "df_baseline": "titanic 1"},
        )
        smart_drift.compile()
        assert isinstance(smart_drift.xpl, shapash.explainer.smart_explainer.SmartExplainer)

    def test_generate_report_fullvalid(self):
        """
        test generate_report() with fullvalidation option specified to True
        """
        smart_drift = SmartDrift(
            self.titanic_df_1, self.titanic_df_2, deployed_model=self.rf, encoding=self.categ_encoding
        )
        smart_drift.compile(full_validation=True)
        infofile = path.join(self.script_path, PROJECT_INFO_PATH)
        smart_drift.generate_report(output_file=REPORT_PATH, project_info_file=infofile)

        assert path.isfile(REPORT_PATH)
        os.remove(REPORT_PATH)

    def test_generate_report_nofullvalid(self):
        """
        test generate_report() with no full validation
        """
        smart_drift = SmartDrift(
            self.titanic_df_1, self.titanic_df_2, deployed_model=self.rf, encoding=self.categ_encoding
        )
        smart_drift.compile()
        infofile = path.join(self.script_path, PROJECT_INFO_PATH)
        smart_drift.generate_report(output_file=REPORT_PATH, project_info_file=infofile)
        assert path.isfile(REPORT_PATH)
        os.remove(REPORT_PATH)

    def test_add_data_modeldrift_1(self):
        """
        test add_data_modeldrift method
        """
        lift_info = {
            "annee": ["2019", "2019"],
            "mois": ["7", "8"],
            "plage_historique": [1, 1],
            "target": ["souscription", "souscription"],
            "type_lift": ["lift10", "lift10"],
            "performance": [3.32, 3.18],
        }
        df_lift = pd.DataFrame(lift_info)
        smart_drift = SmartDrift(
            self.titanic_df_1, self.titanic_df_2, deployed_model=self.rf, encoding=self.categ_encoding
        )
        smart_drift.compile(ignore_cols=["PassengerId"])
        assert isinstance(smart_drift.data_modeldrift, type(None))
        smart_drift.add_data_modeldrift(dataset=df_lift)
        assert isinstance(smart_drift.data_modeldrift, pd.core.frame.DataFrame)

    def test_add_data_modeldrift_2(self):
        """
        test add_data_modeldrift method
        """
        lift_info = {
            "annee": ["2019", "2019"],
            "mois": ["7", "8"],
            "plage_historique": [1, 1],
            "target": ["souscription", "souscription"],
            "type_lift": ["lift10", "lift10"],
            "lift": [3.32, 3.18],
        }
        df_lift = pd.DataFrame(lift_info)
        smart_drift = SmartDrift(
            self.titanic_df_1, self.titanic_df_2, deployed_model=self.rf, encoding=self.categ_encoding
        )
        smart_drift.compile(ignore_cols=["PassengerId"])
        assert isinstance(smart_drift.data_modeldrift, type(None))
        with self.assertRaises(Exception):
            smart_drift.add_data_modeldrift(dataset=df_lift)
        smart_drift.add_data_modeldrift(dataset=df_lift, metric="lift")
        assert isinstance(smart_drift.data_modeldrift, pd.core.frame.DataFrame)
        smart_drift.add_data_modeldrift(
            dataset=df_lift, metric="lift", reference_columns=["target", "plage_historique"]
        )
        assert isinstance(smart_drift.data_modeldrift, pd.core.frame.DataFrame)

    def test_analyse_consistency_nofullvalidation_1(self):
        """
        Test _analyze_consistency() method with full validation defined to False
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent.parent
        titanic_original = path.join(script_path, TITANIC_ORIGINAL_PATH)
        df_original = pd.read_csv(titanic_original, index_col=0)
        titanic_altered = path.join(script_path, TITANIC_ALTERED_PATH)
        df_altered = pd.read_csv(titanic_altered, index_col=0)

        df_altered["col_sup"] = 0
        smart_drift = SmartDrift(df_altered, df_original)
        pb_cols, err_mods = smart_drift._analyze_consistency(
            full_validation=False, ignore_cols=["col_sup", "Name", "PassengerId"]
        )

        expected_pb_cols = {"New columns": [], "Removed columns": ["col_sup"], "Type errors": []}

        assert isinstance(pb_cols, dict)
        assert all(key in ("New columns", "Removed columns", "Type errors") for key in pb_cols.keys())
        assert isinstance(err_mods, dict)
        assert pb_cols == expected_pb_cols
        assert err_mods == dict()

    def test_analyse_consistency_nofullvalidation_2(self):
        """
        Test _analyze_consistency() method with full validation defined to False
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent.parent
        titanic_original = path.join(script_path, TITANIC_ORIGINAL_PATH)
        df_original = pd.read_csv(titanic_original, index_col=0)
        titanic_altered = path.join(script_path, TITANIC_ALTERED_PATH)
        df_altered = pd.read_csv(titanic_altered, index_col=0)

        df_altered["col_sup"] = 0
        smart_drift = SmartDrift(df_altered, df_original)
        pb_cols, err_mods = smart_drift._analyze_consistency(
            full_validation=False, ignore_cols=["col_sup", "Name", "PassengerId"]
        )

        with self.assertRaises(TypeError):
            smart_drift.df_current = dict()
            smart_drift._analyze_consistency(full_validation=False, ignore_cols=["col_sup", "Name", "PassengerId"])

    def test_analyse_consistency_fullvalidation(self):
        """
        Test _analyze_consistency() method with full validation defined to True
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent.parent
        titanic_original = path.join(script_path, TITANIC_ORIGINAL_PATH)
        df_original = pd.read_csv(titanic_original, index_col=0)
        titanic_altered = path.join(script_path, TITANIC_ALTERED_PATH)
        df_altered = pd.read_csv(titanic_altered, index_col=0)

        df_altered["col_sup"] = 0
        df_altered = df_altered.replace("male", "Male")
        df_original["col_sup"] = "0"
        smart_drift = SmartDrift(df_altered, df_original)
        pb_cols, err_mods = smart_drift._analyze_consistency(
            full_validation=True, ignore_cols=["Title", "Name", "PassengerId"]
        )

        expected_err_mods = {"Sex": {"New distinct values": ["Male"], "Removed distinct values": ["male"]}}

        expected_pb_cols = {"New columns": [], "Removed columns": [], "Type errors": ["col_sup"]}

        assert isinstance(pb_cols, dict)
        assert all(key in ("New columns", "Removed columns", "Type errors") for key in pb_cols.keys())
        assert isinstance(err_mods, dict)
        assert pb_cols == expected_pb_cols
        assert err_mods == expected_err_mods

    def test_predict_1(self):
        """
        test _predict() method
        """

        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        df_predict = smart_drift._predict()
        assert df_predict is None

        with self.assertRaises(Exception):
            smart_drift._predict(deployed_model=self.rf)

        df_predict = smart_drift._predict(deployed_model=self.rf, encoding=self.categ_encoding)
        assert isinstance(df_predict, pd.DataFrame)
        assert all(column in ("dataset", "Score") for column in df_predict.columns)
        assert df_predict.shape[0] == (smart_drift.df_baseline.shape[0] + smart_drift.df_current.shape[0])
        assert all(col_type in ("float64", "object") for col_type in df_predict.dtypes)

    def test_predict_2(self):
        """
        test _predict() method
        """

        varcat = ["Pclass", "Sex", "Embarked", "Title"]

        false_categ_encoding = OrdinalEncoder(cols=varcat, handle_unknown="ignore", return_df=True).fit(
            self.X[["Title", "Pclass", "Sex", "Embarked"]]
        )
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)

        with self.assertRaises(Exception):
            smart_drift._predict(deployed_model=self.rf, encoding=false_categ_encoding)

        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        df_predict = smart_drift._predict(deployed_model=self.rf, encoding=self.categ_encoding)

        assert isinstance(df_predict, pd.DataFrame)
        assert all(column in ("dataset", "Score") for column in df_predict.columns)
        assert df_predict.shape[0] == (smart_drift.df_baseline.shape[0] + smart_drift.df_current.shape[0])
        assert all(col_type in ("float64", "object") for col_type in df_predict.dtypes)

    def test_predict_3(self):
        """
        test _predict() method
        """
        varcat = ["Pclass", "Sex", "Embarked", "Title"]
        false_categ_encoding = OrdinalEncoder(cols=varcat, handle_unknown="ignore", return_df=True).fit(
            self.X[["Title", "Pclass", "Sex", "Embarked"]]
        )
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)

        with self.assertRaises(Exception):
            smart_drift._predict(deployed_model=self.rf, encoding=false_categ_encoding)

        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)

        with self.assertRaises(Exception):
            smart_drift._predict(deployed_model=self.rf, encoding=false_categ_encoding)

        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        df_predict = smart_drift._predict(deployed_model=self.rf, encoding=self.categ_encoding)

        assert isinstance(df_predict, pd.DataFrame)
        assert all(column in ("dataset", "Score") for column in df_predict.columns)
        assert df_predict.shape[0] == (smart_drift.df_baseline.shape[0] + smart_drift.df_current.shape[0])
        assert all(col_type in ("float64", "object") for col_type in df_predict.dtypes)

        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        df_predict = smart_drift._predict(deployed_model=self.rf, encoding=self.categ_encoding)
        assert isinstance(df_predict, pd.DataFrame)
        assert all(column in ("dataset", "Score") for column in df_predict.columns)
        assert df_predict.shape[0] == (smart_drift.df_baseline.shape[0] + smart_drift.df_current.shape[0])
        assert all(col_type in ("float64", "object") for col_type in df_predict.dtypes)

    def test_feature_importance(self):
        """
        Test on _feature_importance()'s method
        """
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smart_drift.compile()

        feature_imp = smart_drift._feature_importance()
        assert feature_imp is None

        feature_imp = smart_drift._feature_importance(deployed_model=self.rf)
        assert isinstance(feature_imp, pd.DataFrame)
        assert all(column in ("feature", "deployed_model", "datadrift_classifier") for column in feature_imp.columns)
        assert all(col_type in ("float64", "object") for col_type in feature_imp.dtypes)

    def test_histo_datadrift_metric_1(self):
        """
        Test on _histo_datadrift_metric() method
        """
        SD = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        df_auc = SD._histo_datadrift_metric()
        assert df_auc == None

        SD = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        SD.compile(datadrift_file="tests/data/AUC_Histo.csv")
        df_auc = SD._histo_datadrift_metric()
        assert isinstance(df_auc, pd.DataFrame)
        assert all(column in ("date", "auc") for column in df_auc.columns)
        assert all(column in ("object", "float64") for column in df_auc.dtypes)
        os.remove("tests/data/AUC_Histo.csv")

    def test_histo_datadrift_metric_2(self):
        """
        Test on _histo_datadrift_metric() method
        """
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smart_drift.compile(datadrift_file="tests/data/AUC_Histo.csv")

        with self.assertRaises(Exception):
            smart_drift._histo_datadrift_metric(
                datadrift_file="tests/data/AUC_Histo.csv", predict_test=smart_drift.model.predict(self.titanic_df_2)
            )

        df_auc = smart_drift._histo_datadrift_metric()
        assert isinstance(df_auc, pd.DataFrame)
        assert all(column in ("date", "auc") for column in df_auc.columns)
        assert all(column in ("object", "float64") for column in df_auc.dtypes)
        os.remove("tests/data/AUC_Histo.csv")

    def test_histo_datadrift_metric_3(self):
        """
        Test on _histo_datadrift_metric() method
        """
        smart_drift = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        smart_drift.compile(datadrift_file="tests/data/AUC_Histo.csv")

        with self.assertRaises(Exception):
            smart_drift._histo_datadrift_metric(date_compile_auc="01-12-2020")

        datadrift_file = path.join(self.script_path, "tests/data/AUC_Histo.csv")
        df_auc = smart_drift._histo_datadrift_metric(date_compile_auc="01/12/2020", datadrift_file=datadrift_file)
        assert isinstance(df_auc, pd.DataFrame)
        assert all(column in ("date", "auc") for column in df_auc.columns)
        assert all(column in ("object", "float64") for column in df_auc.dtypes)
        os.remove("tests/data/AUC_Histo.csv")

    def test_histo_datadrift_metric_4(self):
        """
        Test on _histo_datadrift_metric() method
        """
        smart_drift = SmartDrift(
            self.titanic_df_1, self.titanic_df_2, deployed_model=self.rf, encoding=self.categ_encoding
        )
        smart_drift.compile(datadrift_file="tests/data/AUC_Histo.csv")

        with self.assertRaises(Exception):
            smart_drift._histo_datadrift_metric(date_compile_auc="01-12-2020")

        datadrift_file = path.join(self.script_path, "tests/data/AUC_Histo.csv")
        df_auc = smart_drift._histo_datadrift_metric(date_compile_auc="01/12/2020", datadrift_file=datadrift_file)
        assert isinstance(df_auc, pd.DataFrame)
        assert all(column in ("date", "auc", "JS_predict") for column in df_auc.columns)
        assert all(column in ("object", "float64") for column in df_auc.dtypes)
        os.remove("tests/data/AUC_Histo.csv")

    def test_save_load(self):
        """
        Test on save() and load() methods : checks that the attributes meant to be saved and loaded are saved and loaded
        """
        temp_pkl_path = "tests/data/test_pkl.pkl"
        sd = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        sd.compile(datadrift_file="tests/data/AUC_Histo.csv")
        sd.save(temp_pkl_path)
        sd2 = SmartDrift.load(temp_pkl_path)
        assert isinstance(sd2, SmartDrift)
        pd.testing.assert_frame_equal(sd.df_baseline, sd2.df_baseline)
        assert len(sd.df_baseline) == len(sd2.df_baseline)
        self.assertCountEqual(sd.df_current.columns, sd2.df_current.columns)
        assert len(sd.df_current) == len(sd2.df_current)
        assert sd.feature_importance == sd2.feature_importance
        self.assertCountEqual(sd.ignore_cols, sd2.ignore_cols)
        self.assertCountEqual(sd.pb_cols, sd2.pb_cols)
        self.assertCountEqual(sd.err_mods, sd2.err_mods)
        assert sd.auc == sd2.auc
        pd.testing.assert_frame_equal(sd.historical_auc, sd2.historical_auc)
        pd.testing.assert_frame_equal(sd._df_concat, sd2._df_concat)
        assert sd._datadrift_target == sd2._datadrift_target
        assert sd.deployed_model == sd2.deployed_model
        assert sd.encoding == sd2.encoding
        assert sd.palette_name == sd2.palette_name
        assert sd.colors_dict == sd2.colors_dict
        os.remove(temp_pkl_path)

    def test_define_style(self):
        """
        test define_style() method : checks that the frontend style defined in colors.json is written into the SmartDrift plot attribute
        """
        sd = SmartDrift(self.titanic_df_1, self.titanic_df_2)
        sd.compile(datadrift_file="tests/data/AUC_Histo.csv")
        colors_dict = select_palette(colors_loading(), "eurybia")
        with self.assertRaises(ValueError):
            sd.define_style()
        sd.define_style(palette_name="eurybia", colors_dict=colors_dict)
        assert sd.colors_dict == colors_dict
        assert sd.plot._style_dict["univariate_cat_bar"] == colors_dict["univariate_cat_bar"]
        assert sd.plot._style_dict["univariate_cont_bar"] == colors_dict["univariate_cont_bar"]
        assert sd.plot._style_dict["datadrift_historical"] == colors_dict["datadrift_historical"]
        assert sd.plot._style_dict["scatter_plot"] == colors_dict["scatter_plot"]
        assert sd.plot._style_dict["scatter_line"] == colors_dict["scatter_line"]
        assert sd.plot._style_dict["featimportance_colorscale"] == colors_dict["featimportance_colorscale"]
        assert sd.plot._style_dict["contrib_colorscale"] == colors_dict["contrib_colorscale"]
        # not testing the shapash.explainer.smart_explainer

    def test_datetime_column_transformation(self):
        """
        Test if SmartDrift can automatically handle datatime columns
        """

        date_list = pd.date_range(start="01/01/2022", end="01/30/2022")
        X1 = np.random.rand(len(date_list))
        X2 = np.random.rand(len(date_list))

        df_current = pd.DataFrame(date_list, columns=["date"])
        df_current["col1"] = X1
        df_baseline = pd.DataFrame(date_list, columns=["date"])
        df_baseline["col1"] = X2

        sd = SmartDrift(df_current=df_current, df_baseline=df_baseline)
        sd.compile(full_validation=True)
        # Should pass this step
        auc = sd.auc
        assert auc > 0

    def test_datetime_column_model_error(self):
        """
        Test if SmartDrift raised an error when their is datatime columns
        and deployed_model is filled
        """

        date_list = pd.date_range(start="01/01/2022", end="01/30/2022")
        X1 = np.random.rand(len(date_list))
        X2 = np.random.rand(len(date_list))

        df_current = pd.DataFrame(date_list, columns=["date"])
        df_current["col1"] = X1
        df_baseline = pd.DataFrame(date_list, columns=["date"])
        df_baseline["col1"] = X2

        # Random models
        regressor = RandomForestRegressor(n_estimators=2).fit(df_baseline[["col1"]], df_baseline["col1"].ravel())

        sd = SmartDrift(df_current=df_current, df_baseline=df_baseline, deployed_model=regressor)

        # Should raise an error
        with pytest.raises(TypeError, match="df_current have datetime column. You should drop it"):
            sd.compile(full_validation=True)
