"""
- SmartDrift module
"""

import copy
import datetime
import io
import logging
import pickle
import shutil
import tempfile
from pathlib import Path

import catboost
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from eurybia.core.smartplotter import SmartPlotter
from eurybia.style.style_utils import colors_loading, select_palette
from eurybia.utils.io import load_pickle, save_pickle
from eurybia.utils.model_drift import catboost_hyperparameter_init, catboost_hyperparameter_type
from eurybia.utils.statistical_tests import chisq_test, compute_js_divergence, ksmirnov_test
from eurybia.utils.utils import base_100, convert_date_col_into_multiple_col

logging.getLogger("papermill").setLevel(logging.WARNING)
logging.getLogger("blib2to3").setLevel(logging.WARNING)


class SmartDrift:
    """
    The SmartDrift class is the main object to compute drift in the Eurybia library
    It allows to calculate data drift between 2 datasets using a data drift classification model

    Attributes
    ----------
    df_current: pandas.DataFrame
        current (or production) dataset which is compared to df_baseline
    df_baseline: pandas.DataFrame
        baseline (or learning) dataset which is compared to df_current
    datadrift_classifier: model object
        model used for binary classification of data drift
    xpl: Shapash object
        object used to compute explainability on datadrift_classifier
    df_predict: pandas.DataFrame
        computed score on both datasets if a deployed_model is specified
    feature_importance: pandas.DataFrame
        feature importance of datadrift_classifier and feature importance of production model if exist
    pb_cols: dict
        Dictionnary that references columns differences between df_current and df_baseline
    err_mods: dict
        Dictionnary that references modalities differences in columns between df_current and df_baseline
    auc: int
        Value auc of model drift
    historical_auc: pandas.DataFrame
        Dataframe that contains auc history of datadrift_classifier over time
    data_modeldrift: pandas.DataFrame
        Dataframe that contains performance history of deployed_model
    ignore_cols: list
        list of feature to ignore in compute
    dataset_names : dict, (Optional)
        Dictionnary used to specify dataset names to display in report.
    _df_concat : pandas.DataFrame
        Dataframe that's composed of both df_baseline and df_current concatenated
    plot : eurybia.core.smartplotter.SmartPlotter
        Instance of an Eurybia SmartPlotter class. It's used for graph displaying purpose.
    deployed_model: model object, optional
            model in production used to put in perspective drift and to predict
    encoding: preprocessing object, optional (default: None)
            Preprocessing used before the training step
    datadrift_stat_test : dict
        Datadrift statistical tests for each feature.
        Each test identifies whether the feature has drifted.
        There are 2 types of test implemented depending on the type of feature:
        - Chi-square for discrete variables - ref:
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html)
        - Kolmogorov-Smirnov for continuous variables - ref:
        (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html)
        This datadrift_stat_test attribute specifies for each feature the test performed,
        the statistic the test and the p value
    palette_name : str (default: 'eurybia')
        Name of the palette used for the colors of the report (refer to style folder).
    colors_dict: dict
            Dict of the colors used in the different plots
    datadrift_file : str, optional
            Name of the csv file that contains the performance history of data drift
            If no datadrift file is given, the drift will not be logged
    js_divergence : float
        Jensen-Shannon divergence of probability distributions - ref:
        (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)

    How to declare a new SmartDrift object?

    Example
    --------
    >>> SD = Smartdrift(df_current=df_production, df_baseline=df_learning)

    """

    @classmethod
    def load(cls, path):
        """
        The load() class method allows Eurybia users to use a pickled SmartDrift.

        Parameters
        ----------
        path : str
            File path of the pickle file.
        Returns
        ----------
        SmartDrift
            SmartDrift instance loaded with the pickle given as "path"
        Example
        --------
        >>> from eurybia import SmartDrift
        >>> SmartDrift.load('path_to_pkl/smardrift.pkl')
        """
        dict_to_load = load_pickle(path)
        sd = cls()
        if isinstance(dict_to_load, dict):
            for attr, val in dict_to_load.items():
                if isinstance(val, io.BytesIO):
                    setattr(sd, attr, pickle.load(val.seek(0)))
                elif attr == "xpl":
                    xpl = SmartExplainer(model=val["model"])
                    xpl.__dict__.update(val)
                    setattr(sd, attr, xpl)
                else:
                    setattr(sd, attr, val)
        else:
            raise ValueError("pickle file must contain dictionary")
        return sd

    def __init__(
        self,
        df_current=None,
        df_baseline=None,
        dataset_names={"df_current": "Current dataset", "df_baseline": "Baseline dataset"},
        deployed_model=None,
        encoding=None,
        palette_name="eurybia",
        colors_dict=None,
    ):
        """
        Parameters
        ----------
        df_current: pandas.DataFrame
            current (or production) dataset which is compared to df_baseline
        df_baseline: pandas.DataFrame
            baseline (or learning) dataset which is compared to df_current
        dataset_names : dict, (Optional)
            Dictionnary used to specify dataset names to display in report.
        deployed_model: model object, optional
                model in production used to put in perspective drift and to predict
        encoding: preprocessing object, optional (default: None)
                Preprocessing used before the training step
        palette_name : str (default: 'eurybia')
            Name of the palette used for the colors of the report (refer to style folder).
        colors_dict: dict
                Dict of the colors used in the different plots

        How to declare a new SmartDrift object ?

        Example
        --------

        >>> SD = Smartdrift(df_current=df_production, df_baseline=df_learning)
        """
        self.df_current = df_current
        self.df_baseline = df_baseline
        self.xpl = None
        self.df_predict = None
        self.feature_importance = None
        self.pb_cols, self.err_mods = None, None
        self.auc = None
        self.js_divergence = None
        self.historical_auc = None
        self.data_modeldrift = None
        self.ignore_cols = list()
        self.datadrift_stat_test = None
        if "df_current" not in dataset_names.keys() or "df_baseline" not in dataset_names.keys():
            raise ValueError("dataset_names must be a dictionnary with keys 'df_current' and 'df_baseline'")
        self.dataset_names = pd.DataFrame(dataset_names, index=[0])
        self._df_concat = None
        self._datadrift_target = "target"
        self.plot = SmartPlotter(self)
        self.deployed_model = deployed_model
        self.encoding = encoding
        self.palette_name = palette_name
        self.colors_dict = copy.deepcopy(select_palette(colors_loading(), self.palette_name))
        if colors_dict is not None:
            self.colors_dict.update(colors_dict)
        self.plot.define_style_attributes(colors_dict=self.colors_dict)
        self.datadrift_file = None

    def compile(
        self,
        full_validation=False,
        ignore_cols: list = None,
        sampling=True,
        sample_size=100000,
        datadrift_file=None,
        date_compile_auc=None,
        hyperparameter: dict = catboost_hyperparameter_init.copy(),
        attr_importance="feature_importances_",
    ):
        r"""
        The compile method is the first step to compute data drift.
        It allows to calculate data drift between 2 datasets using a data drift classification model.
        Most of the parameters are optional but helps to adapt the data drift calculation if necessary.
        This step can last a few moments with large datasets.

        Parameters
        ----------
        full_validation: bool, optional (default: False)
            If True, analyze consistency on modalities between columns
        ignore_cols: list, optional
            list of feature to ignore in compute
        sampling: bool, optional
            If True, applies the sampling
        sample_size: int, optional
            the size of the sample to build
        date_compile_auc: str (optional)
            format dd/mm/yyyy use for specify date of compute drift, useful when compute few time drift for different time at the same moment
        hyperparameter: dict, optional
            if user want to modify catboost hyperparameter
        attr_importance: string, optional (default: "feature_importances\_")
            Attribute "feature_importance" of the deployed_model
        datadrift_file : str, optional
            Name of the csv file that contains the performance history of data drift. If no datadrift file is given, the drift will not be logged

        Examples
        --------
        >>> SD.compile()

        """
        if ignore_cols is None:
            ignore_cols = []
        if datadrift_file is not None:
            self.datadrift_file = datadrift_file
        if hyperparameter is not None:
            for key, value in catboost_hyperparameter_init.items():
                catboost_hyperparameter_init[key] = (
                    hyperparameter[key]
                    if key in hyperparameter and str(type(hyperparameter[key])) in catboost_hyperparameter_type[key]
                    else value
                )
        hyperparameter = catboost_hyperparameter_init.copy()
        if sample_size is not None:
            self.df_baseline = self._sampling(sampling, sample_size, self.df_baseline)
            self.df_current = self._sampling(sampling, sample_size, self.df_current)

        # Checking datasets
        self._check_dataset(ignore_cols)
        # Consistency analysis
        pb_cols, err_mods = self._analyze_consistency(full_validation, ignore_cols)

        # Adding results to ignored columns
        ignore_cols = list(set(ignore_cols + [item for sublist in [pb_cols[sl] for sl in pb_cols] for item in sublist]))
        if len(ignore_cols) != 0:
            self.df_baseline = self.df_baseline[[c for c in self.df_baseline.columns if c not in ignore_cols]]
            self.df_current = self.df_current[[c for c in self.df_current.columns if c not in ignore_cols]]

        df_concat = (
            pd.concat([self.df_current, self.df_baseline], keys=[1, 0])
            .reset_index()
            .rename(columns={"level_0": self._datadrift_target})
        )
        df_concat.drop(df_concat.columns[1], axis=1, inplace=True)
        varz = [c for c in df_concat.columns if c not in [self._datadrift_target] and c not in ignore_cols]
        dtypes = df_concat[varz].dtypes.map(str)
        cat_features = list(dtypes[dtypes.isin(["object"])].index)
        df_concat[cat_features] = df_concat[cat_features].fillna("NA")
        self._df_concat = df_concat

        train, test = train_test_split(df_concat[varz + [self._datadrift_target]], test_size=0.25, random_state=42)

        i = 0
        indice_cat = []
        for var_x in df_concat[varz]:
            if var_x in cat_features:
                indice_cat.append(i)
            i = i + 1
        train_pool_cat = catboost.Pool(data=train[varz], label=train["target"].astype(int), cat_features=indice_cat)
        test_pool_cat = catboost.Pool(data=test[varz], label=test["target"].astype(int), cat_features=indice_cat)
        datadrift_classifier = catboost.CatBoostClassifier(
            max_depth=hyperparameter["max_depth"],
            l2_leaf_reg=hyperparameter["l2_leaf_reg"],
            learning_rate=hyperparameter["learning_rate"],
            iterations=hyperparameter["iterations"],
            use_best_model=hyperparameter["use_best_model"],
            custom_loss=hyperparameter["custom_loss"],
            loss_function=hyperparameter["loss_function"],
            eval_metric=hyperparameter["eval_metric"],
            task_type="CPU",
            allow_writing_files=False,
        )

        datadrift_classifier = datadrift_classifier.fit(train_pool_cat, eval_set=test_pool_cat, silent=True)

        xpl = SmartExplainer(
            label_dict={0: self.dataset_names["df_baseline"].values[0], 1: self.dataset_names["df_current"].values[0]},
            model=datadrift_classifier,
        )

        x_test = test[varz]
        y_test = test[self._datadrift_target]

        xpl.compile(x=x_test)
        xpl.compute_features_import(force=True)

        self.xpl = xpl
        self.xpl.define_style(colors_dict=self.colors_dict)
        self.datadrift_classifier = datadrift_classifier
        self.df_predict = self._predict(deployed_model=self.deployed_model, encoding=self.encoding)
        self.auc = roc_auc_score(y_test, datadrift_classifier.predict(x_test))
        self.feature_importance = self._feature_importance(
            deployed_model=self.deployed_model, attr_importance=attr_importance
        )
        self.plot.feature_importance = self.feature_importance
        self.pb_cols, self.err_mods = pb_cols, err_mods
        if self.deployed_model is not None:
            self.js_divergence = compute_js_divergence(
                self.df_predict.loc[lambda df: df["dataset"] == self.dataset_names["df_baseline"].values[0], :][
                    "Score"
                ].values,
                self.df_predict.loc[lambda df: df["dataset"] == self.dataset_names["df_current"].values[0], :][
                    "Score"
                ].values,
                n_bins=20,
            )

        self.historical_auc = self._histo_datadrift_metric(
            datadrift_file=self.datadrift_file, date_compile_auc=date_compile_auc
        )
        self.data_modeldrift = None
        self.ignore_cols = ignore_cols
        if self.deployed_model is not None:
            self.datadrift_stat_test = self._compute_datadrift_stat_test()

    def generate_report(
        self, output_file, project_info_file=None, title_story="Drift Report", title_description="", working_dir=None
    ):
        """
        This method will generate an HTML report containing different information about the project.
        It allows the information compiled to be rendered.
        It can be associated with a project info yml file on which can figure different information about the project.

        Parameters
        ----------
        output_file : str
            Path to the HTML file to write
        project_info_file : str
            Path to the file used to display some information about the project in the report
        title_story : str, optional
            Report title
        title_description : str, optional
            Report title description (as written just below the title)
        working_dir : str, optional
            Working directory in which will be generated the notebook used to create the report and where the objects used to execute it will
            be saved. This parameter can be usefull if one wants to create its own custom report and debug the notebook used to generate
            the html report. If None, a temporary directory will be used

        Examples
        --------
        >>> SD.generate_report(
                output_file='report.html',
                project_info_file='project_info.yml',
                title_story="Drift project report",
                title_description="This document is a drift report of the score in production"
            )
        """
        from eurybia.report.generation import execute_report

        rm_working_dir = False
        if not working_dir:
            working_dir = tempfile.mkdtemp()
            rm_working_dir = True

        try:
            execute_report(
                project_info_file=project_info_file,
                explainer=self.xpl,
                smartdrift=self,
                config_report=dict(title_story=title_story, title_description=title_description),
                output_file=output_file,
            )
        finally:
            if rm_working_dir:
                shutil.rmtree(working_dir)

    def _check_dataset(self, ignore_cols: list = list()):
        """
        Method to check if datasets are correct before to be analysed and if
        it's not, try to modify them and informs the user. In worse case raise
        an error.

        Parameters
        ----------
        full_validation : bool, optional (default: False)
            If True, analyze consistency on modalities between columns
        ignore_cols: list, optional
            list of feature to ignore in compute
        """

        if len([column for column in self.df_current.columns if is_datetime(self.df_current[column])]) > 0:
            if self.deployed_model is None:
                for col in [column for column in self.df_current.columns if is_datetime(self.df_current[column])]:
                    print(
                        f"""Column {col} will be dropped and transformed in df_current by : {col}_year, {col}_month, {col}_day"""
                    )
                self.df_current = convert_date_col_into_multiple_col(self.df_current)
            else:
                raise TypeError("df_current have datetime column. You should drop it")

        if len([column for column in self.df_baseline.columns if is_datetime(self.df_baseline[column])]) > 0:
            if self.deployed_model is None:
                for col in [column for column in self.df_baseline.columns if is_datetime(self.df_baseline[column])]:
                    print(
                        f"""Column {col} will be dropped and transformed in df_baseline by : {col}_year, {col}_month, {col}_day"""
                    )
                self.df_baseline = convert_date_col_into_multiple_col(self.df_baseline)
            else:
                raise TypeError("df_baseline have datetime column. You should drop it")

    def _analyze_consistency(self, full_validation=False, ignore_cols: list = list()):
        """
        method to analyse consistency between the 2 datasets, in terms of columns and modalities

        Parameters
        ----------
        full_validation : bool, optional (default: False)
            If True, analyze consistency on modalities between columns
        ignore_cols: list, optional
            list of feature to ignore in compute
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s: %(message)s", datefmt="%y/%m/%d %H:%M:%S"
        )

        if not isinstance(self.df_current, pd.DataFrame) or not isinstance(self.df_baseline, pd.DataFrame):
            raise TypeError("df_current and df_baseline should be Pandas dataframes.")
        if len(ignore_cols) > 0:
            print(f"""The following variables are manually set to be ignored in the analysis: \n {ignore_cols}""")
        # Features
        new_cols = [c for c in self.df_baseline.columns if c not in self.df_current.columns]
        removed_cols = [c for c in self.df_current.columns if c not in self.df_baseline.columns]
        if len(new_cols) > 0:
            print(
                f"""The following variables are no longer available in the
                        current dataset and will not be analyzed: \n {new_cols}"""
            )
        if len(removed_cols) > 0:
            print(
                f"""The following variables are only available in the
            current dataset and will not be analyzed: \n {removed_cols}"""
            )
        common_cols = [c for c in self.df_current.columns if c in self.df_baseline.columns]
        # dtypes
        err_dtypes = [
            c for c in common_cols if self.df_baseline.dtypes.map(str)[c] != self.df_current.dtypes.map(str)[c]
        ]

        if len(err_dtypes) > 0:
            print(
                f"""The following variables have mismatching dtypes
             and will not be analyzed: \n {err_dtypes}"""
            )
        # Feature values
        err_mods: dict[str, dict] = {}
        if full_validation is True:
            invalid_cols = ignore_cols + new_cols + removed_cols + err_dtypes
            for column in self.df_baseline.columns:
                if column not in invalid_cols and self.df_baseline.dtypes.map(str)[column] == "object":
                    uniques_histo = pd.unique(self.df_baseline[column])
                    uniques_current = pd.unique(self.df_current[column])
                    new_mods = [mod for mod in uniques_current if mod not in uniques_histo]
                    removed_mods = [mod for mod in uniques_histo if mod not in uniques_current]
                    if len(new_mods) > 0 or len(removed_mods) > 0:
                        err_mods[column] = {}
                        err_mods[column]["New distinct values"] = new_mods
                        err_mods[column]["Removed distinct values"] = removed_mods
                        print(
                            f"""The variable {column} has mismatching unique values:
{new_mods} | {removed_mods}\n"""
                        )
        return ({"New columns": new_cols, "Removed columns": removed_cols, "Type errors": err_dtypes}, err_mods)

    def _predict(self, deployed_model=None, encoding=None):
        """
        Create an attributes df_predict with the computed score on both datasets

        Parameters
        ----------
        deployed_model : model object, optional (default: None)
            model in production used to put in perspective drift and to predict
        encoding : preprocessing object, optional (default: None)
            Preprocessing used before the training step
        Returns
        -------
        pandas.DataFrame, None
            DataFrame with predicted score for both datasets
        """
        if deployed_model is None:
            return None
        if not hasattr(deployed_model, "predict_proba") and not hasattr(deployed_model, "predict"):
            raise Exception("deployed_model need to have predict or predict_proba method")
        df_baseline = self.df_baseline
        df_current = self.df_current
        if encoding is not None:
            try:
                df_baseline = encoding.transform(df_baseline)
                df_current = encoding.transform(df_current)
            except BaseException as error:
                raise Exception(
                    """
                    Encoding specified can't be applied directly on df_current/df_baseline
                    - Error :
                                    """
                    + str(error)
                )
        if hasattr(deployed_model, "predict_proba"):
            try:
                df_baseline_pred = pd.DataFrame(deployed_model.predict_proba(df_baseline)[:, 1], columns=["Score"])
                df_current_pred = pd.DataFrame(deployed_model.predict_proba(df_current)[:, 1], columns=["Score"])
            except BaseException as error:
                raise Exception(
                    """
                    Encoding specified or deployed_model used can't be applied directly on df_current/df_baseline
                    - Error :
                                    """
                    + str(error)
                )
        else:
            try:
                df_baseline_pred = pd.DataFrame(deployed_model.predict(df_baseline), columns=["Score"])
                df_current_pred = pd.DataFrame(deployed_model.predict(df_current), columns=["Score"])
            except BaseException as error:
                raise Exception(
                    """
                    Encoding specified or deployed_model used can't be applied directly on df_current/df_baseline
                    - Error :
                                    """
                    + str(error)
                )
        return pd.concat(
            [
                df_baseline_pred.assign(dataset=self.dataset_names["df_baseline"].values[0]),
                df_current_pred.assign(dataset=self.dataset_names["df_current"].values[0]),
            ]
        ).reset_index(drop=True)

    def _feature_importance(self, deployed_model=None, attr_importance="feature_importances_"):
        """
        Create an attributes feature_importance with the computed score on both datasets

        Parameters
        ----------
        deployed_model : model object, optional (default: None)
            model in production used to put in perspective drift and to predict
        attr_importance : string, optional (default: "feature_importances_")
            Attribute "feature_importance" of the deployed_model
        Returns
        -------
        pandas.DataFrame, None
            DataFrame with feature importance from production model
            and drift model.
        """
        if deployed_model is None:
            return None
        try:
            array_importance = getattr(deployed_model, attr_importance)
        except BaseException as error:
            raise Exception(
                """
            deployed_model used can't allow to get features importance on df_baseline
            - Error :
                            """
                + str(error)
            )
        feature_importance_drift = pd.DataFrame(
            self.xpl.features_imp[0].values, index=self.xpl.features_imp[0].index, columns=["datadrift_classifier"]
        )
        var_baseline = [c for c in self.df_baseline.columns if c not in ["target"]]
        if len(array_importance) != len(var_baseline):
            raise ValueError(
                """
            Number of features in df_baseline doesn't match feature importance's shape returned by deployed model.
            """
            )
        feature_importance_model_prod = pd.DataFrame(
            array_importance, index=self.df_baseline[var_baseline].columns, columns=["deployed_model"]
        )
        feature_importance = feature_importance_model_prod.merge(
            feature_importance_drift, how="left", left_index=True, right_index=True
        ).reset_index()
        feature_importance = feature_importance.rename(columns={"index": "feature"})
        feature_importance["deployed_model"] = base_100(feature_importance["deployed_model"])
        return feature_importance

    def _sampling(self, sampling, sample_size, dataset):
        """
        Return a sampling from the original dataframe

        Parameters
        ----------
        sampling : bool
            If True, applies the sampling
        sample_size : int
            the size of the sample to build
        df : pd.DataFrame
            The Dataframe to apply sampling
        Returns
        -------
        pandas.DataFrame
            a sample of the original DataFrame or the original DataFrame
        """
        if sampling:
            if dataset.shape[0] > sample_size:
                return dataset.sample(sample_size)
            else:
                return dataset
        else:
            return dataset

    def _histo_datadrift_metric(self, datadrift_file=None, date_compile_auc=None):
        """
        Method which computes datadrift metrics (AUC, and Jensen Shannon prediction divergence if the deployed_model is filled in)
        and append it into a dataframe that will be exported during the generate_report method

        Parameters
        ----------
        datadrift_file : str, (optional)
        date_compile_auc: str (optional)
            format dd/mm/yyyy use for specify date of compute drift, useful when compute few time drift for different time at the same moment
        Returns
        -------
        pandas.DataFrame or None
        Dataframe with dates, AUC and Jensen Shannon prediction divergence computed at this date
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s: %(message)s", datefmt="%y/%m/%d %H:%M:%S"
        )
        if self.datadrift_file is None and date_compile_auc is None:
            return None
        elif self.datadrift_file is None and date_compile_auc is not None:
            self.datadrift_file = "historical_AUC.csv"
            return None
        else:
            if date_compile_auc:
                try:
                    datetime.datetime.strptime(date_compile_auc, "%d/%m/%Y")
                    # FIXME find exact exception type and use it instead of "Exception"
                except Exception:
                    raise Exception("The argument date must have the format '%d/%m/%Y'")
            else:
                date_compile_auc = (datetime.datetime.today().date()).strftime("%d/%m/%Y")
            print(f"The computed AUC on the X_test used to build datadrift_classifier is equal to: {self.auc}")

            if self.deployed_model is not None:
                df_auc = pd.DataFrame(
                    {"date": [date_compile_auc], "auc": [self.auc], "JS_predict": [self.js_divergence]}
                )
            else:
                df_auc = pd.DataFrame({"date": [date_compile_auc], "auc": [self.auc]})

            if self.datadrift_file is not None:
                if Path(self.datadrift_file).is_file() and self.datadrift_file.endswith(".csv"):
                    histo_auc = pd.read_csv(self.datadrift_file).reset_index(drop=True)
                    if self.deployed_model is not None:
                        if not (
                            any(histo_auc.columns.isin(["date"]))
                            and any(histo_auc.columns.isin(["auc"]))
                            and any(histo_auc.columns.isin(["JS_predict"]))
                        ):
                            raise Exception("The csv data must have columns 'date', 'auc' and 'JS_predict'")
                        df_auc = pd.concat([histo_auc[["date", "auc", "JS_predict"]], df_auc]).reset_index(drop=True)

                    else:
                        if not (any(histo_auc.columns.isin(["date"])) and any(histo_auc.columns.isin(["auc"]))):
                            raise Exception("The csv data must have columns 'date' and 'auc'")
                        df_auc = pd.concat([histo_auc[["date", "auc"]], df_auc]).reset_index(drop=True)

                else:
                    print(f"{self.datadrift_file} did not exist and was created. ")

                try:
                    df_auc.to_csv(self.datadrift_file)
                except OSError as error:
                    raise OSError("Can't save to csv the AUC metrics, error : " + str(error))
        return df_auc

    def add_data_modeldrift(
        self, dataset, metric="performance", reference_columns=[], year_col="annee", month_col="mois"
    ):
        """
        When method drift is specified, It will display in the report
        the several plots from a dataframe to analyse drift model from the deployed model.
        Each plot will represent one possible computed metric according
        to its groups. (grouped by date(year-month), reference_columns).

        Parameters
        ----------
        df : pd.DataFrame
            The Dataframe with all the computed metrics.
        metric: str, (default: 'performance')
            The column name of the metric computed
        reference_columns: list, (default: [])
            the column names to use for aggregation with the Date computed
        year_col: str, (default: 'annee')
            The column name of the year where the metric has been computed
        month_col: str, (default: 'mois')
            The column name of the month where the metric has been computed
        """
        try:
            df_modeldrift = dataset.copy()
            df_modeldrift[month_col] = df_modeldrift[month_col].apply(lambda row: str(row).split(".")[0])
            df_modeldrift["Date"] = (
                "01/" + df_modeldrift[month_col] + "/" + df_modeldrift[year_col].astype("int64").astype(str)
            )
            df_modeldrift["Date"] = pd.to_datetime(df_modeldrift["Date"], format="%d/%m/%Y")
            df_aggregate = pd.DataFrame(
                df_modeldrift.groupby(["Date"] + reference_columns)[metric].mean()
            ).reset_index()
            df_aggregate["Date"] = pd.to_datetime(df_aggregate["Date"]).dt.strftime("%d/%m/%Y")
            self.data_modeldrift = df_aggregate

        except BaseException as error:
            raise Exception(
                """
            The df specified in the method doesn't allow us to aggregate it for the report.
            - Error -
                            """
                + str(error)
            )

    def _compute_datadrift_stat_test(self, max_size=50000, categ_max=20):
        """
        calculates all statistical tests to analyze the drift of each feature

        Parameters
        ----------
        max_size : int
            Sets the maximum number of rows. If the datasets are larger there is sampling
        categ_max: int
            Maximum number of values \u200b\u200bper feature to apply the chi square test

        Returns :
        -------
        dict :
            keys - features
            values - dict containing testname, statistic, pvalue
        """
        # sampling
        baseline = self.df_baseline.sample(n=max_size) if self.df_baseline.shape[0] > max_size else self.df_baseline
        current = self.df_current.sample(n=max_size) if self.df_current.shape[0] > max_size else self.df_current
        test_results = {}

        # compute test for each feature
        for features, count in self.xpl.features_desc.items():
            try:
                if current[features].dtypes.kind == "O" and count <= categ_max:
                    test = chisq_test(current[features].to_numpy(), baseline[features].to_numpy())
                else:
                    test = ksmirnov_test(current[features].to_numpy(), baseline[features].to_numpy())
            except BaseException as e:
                raise Exception(
                    f"""
                There is a problem with the format of {str(features)} column between the two datasets.
                Error:
                """
                    + str(e)
                )
            test_results[features] = test

        return pd.DataFrame.from_dict(test_results, orient="index")

    def define_style(self, palette_name=None, colors_dict=None):
        """
        the define_style function is a function that uses a palette or a dict
        to define the different styles used in the different outputs
        of eurybia

        Parameters
        ----------
        palette_name : str (default: 'eurybia')
            Name of the palette used for the colors of the report (refer to style folder).
        colors_dict: dict
            Dict of the colors used in the different plots
        """
        if palette_name is None and colors_dict is None:
            raise ValueError("At least one of palette_name or colors_dict parameters must be defined")
        new_palette_name = palette_name or self.palette_name
        new_colors_dict = copy.deepcopy(select_palette(colors_loading(), new_palette_name))
        if colors_dict is not None:
            new_colors_dict.update(colors_dict)
        self.colors_dict.update(new_colors_dict)
        self.plot.define_style_attributes(colors_dict=self.colors_dict)
        self.xpl.define_style(colors_dict=self.colors_dict)

    def save(self, path):
        """
        Save method allows user to save SmartDrift object on disk
        using a pickle file.
        Save method can be useful: you don't have to recompile to display
        results later

        Parameters
        ----------
        path : str
            File path to store the pickle file

        Example
        --------
        >>> smartdrift.save('path_to_pkl/smartdrift.pkl')
        """
        dict_to_save = {}
        for att in self.__dict__.keys():
            if isinstance(getattr(self, att), (list, dict, pd.DataFrame, pd.Series, type(None), bool, float)):
                dict_to_save.update({att: getattr(self, att)})
            elif isinstance(getattr(self, att), SmartExplainer):
                smartexplainer_dict = {}
                for att_xpl in self.xpl.__dict__.keys():
                    if isinstance(
                        getattr(self.xpl, att_xpl), (list, dict, pd.DataFrame, pd.Series, type(None), bool)
                    ) or att_xpl in ["model", "preprocessing", "postprocessing"]:
                        smartexplainer_dict.update({att_xpl: getattr(self.xpl, att_xpl)})
                dict_to_save.update({att: smartexplainer_dict})
        save_pickle(dict_to_save, path)
