"""SmartDrift module"""

import copy
import io
import pickle
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.metrics import roc_auc_score

from eurybia.core.dataset_analysis import DatasetAnalysis
from eurybia.core.smartplotter import SmartPlotter
from eurybia.report.generation import execute_report
from eurybia.style.style_utils import colors_loading, select_palette
from eurybia.utils.io import load_pickle, save_pickle
from eurybia.utils.model_drift import catboost_hyperparameter_init, catboost_hyperparameter_type
from eurybia.utils.statistical_tests import chisq_test, compute_js_divergence, ksmirnov_test
from eurybia.utils.utils import base_100, cat_features_indices, train_test_split_concat


class SmartDrift:
    """The SmartDrift class is the main object to compute drift in the Eurybia library
    It allows to calculate data drift between 2 datasets using a data drift classification model

    Attributes:
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
    df_concat : pandas.DataFrame
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
    js_divergence : float
        Jensen-Shannon divergence of probability distributions - ref:
        (https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)

    How to declare a new SmartDrift object?

    Example:
    -------
    >>> SD = Smartdrift(df_current=df_production, df_baseline=df_learning)

    """

    @classmethod
    def load(cls, path):
        """The load() class method allows Eurybia users to use a pickled SmartDrift.

        Parameters
        ----------
        path : str
            File path of the pickle file.

        Returns
        -------
        SmartDrift
            SmartDrift instance loaded with the pickle given as "path"
        Example
        --------
        >>> from eurybia import SmartDrift
        >>> SmartDrift.load('path_to_pkl/smardrift.pkl')

        """
        dict_to_load = load_pickle(path)
        if isinstance(dict_to_load, dict):
            df_current = dict_to_load["_df_current"]
            df_baseline = dict_to_load["_df_baseline"]
            sd = cls(df_current, df_baseline)

            for attr, val in dict_to_load.items():
                if isinstance(val, io.BytesIO):
                    setattr(sd, attr, pickle.load(val.seek(0)))
                elif attr == "xpl":
                    xpl = SmartExplainer(model=val["model"])
                    xpl.__dict__.update(val)
                    setattr(sd, attr, xpl)
                elif attr == "_da":
                    da = DatasetAnalysis(df_current, df_baseline)
                    da.__dict__.update(val)
                    setattr(sd, attr, da)
                else:
                    setattr(sd, attr, val)
        else:
            raise ValueError("pickle file must contain dictionary")
        return sd

    # FIXME: we should explicitly declare the type of supported deployed_model and encoding
    def __init__(
        self,
        df_current: pd.DataFrame,
        df_baseline: pd.DataFrame,
        dataset_names: tuple[str, str] = ("Current", "Baseline"),
        deployed_model: Any | None = None,
        encoding: Any = None,
        palette_name: str = "eurybia",
        colors_dict: dict | None = None,
    ):
        """Parameters
        ----------
        df_current: pandas.DataFrame
            current (or production) dataset which is compared to df_baseline
        df_baseline: pandas.DataFrame
            baseline (or learning) dataset which is compared to df_current
        dataset_names : tuple, (Optional)
            Tuple used to specify dataset names to display in report (df_current_name, df_baseline_name).
        deployed_model: model object, optional
                model in production used to put in perspective drift and to predict
        encoding: preprocessing object, optional (default: None)
                Preprocessing used before the training step
        palette_name : str (default: 'eurybia')
            Name of the palette used for the colors of the report (refer to style folder).
        colors_dict: dict
                Dict of the colors used in the different plots

        How to declare a new SmartDrift object ?

        Example:
        -------
        >>> SD = Smartdrift(df_current=df_production, df_baseline=df_learning)

        """
        self._df_current = df_current
        self._df_baseline = df_baseline
        self._dataset_names = dataset_names

        # model drift
        self._deployed_model = deployed_model
        self._encoding = encoding

        # report
        self.palette_name = palette_name
        self.colors_dict = copy.deepcopy(select_palette(colors_loading(), self.palette_name))
        if colors_dict is not None:
            self.colors_dict.update(colors_dict)

        # data drift
        self._da: DatasetAnalysis
        self._xpl: SmartExplainer
        self._df_predict: pd.DataFrame
        self._feature_importance: pd.DataFrame

        self._auc: float
        self._js_divergence: float
        self._historical_auc: pd.DataFrame
        self._data_modeldrift: pd.DataFrame

        self._datadrift_stat_test: pd.DataFrame  # smartplotter
        self._datadrift_target: str = "target"  # constant

        self._plot = SmartPlotter(self)
        self._plot.define_style_attributes(colors_dict=self.colors_dict)

        self._modalities_analysis: bool = False

    def compile(
        self,
        full_validation: bool = False,
        ignore_cols: list[str] | None = None,
        sampling: bool = True,
        sample_size: int = 100000,
        datadrift_file: str | None = None,
        date_compile_auc: date | None = None,
        hyperparameter: dict | None = None,
        attr_importance: str = "feature_importances_",
    ):
        r"""The compile method is the first step to compute data drift.
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
        date_compile_auc: date (optional)
            used to specify date of compute drift, useful when compute few time drift
            for different time at the same moment
        hyperparameter: dict, optional
            if user want to modify catboost hyperparameter
        attr_importance: string, optional (default: "feature_importances\_")
            Attribute "feature_importance" of the deployed_model
        datadrift_file : str, optional
            Name of the csv file that contains the performance history of data drift. If no datadrift file is given,
            the drift will not be logged

        Examples
        --------
        >>> SD.compile()

        """
        self._modalities_analysis = full_validation

        if hyperparameter is not None:
            for key, value in catboost_hyperparameter_init.items():
                catboost_hyperparameter_init[key] = (
                    hyperparameter[key]
                    if key in hyperparameter and str(type(hyperparameter[key])) in catboost_hyperparameter_type[key]
                    else value
                )
        hyperparameter = catboost_hyperparameter_init.copy()

        da_sample_size = sample_size if sampling else None

        self.da = DatasetAnalysis(
            df_baseline=self._df_baseline,
            df_test=self._df_current,
            sample_size=da_sample_size,
            ignored_cols=ignore_cols,
        )

        # Checking datasets
        if (len(self.da.datetime_cols) > 0) and (self.deployed_model is not None):
            raise TypeError("Your datasets have a datetime column. You should drop it")

        self.df_current, self.df_baseline = self.da.clean_datasets()

        train, test = train_test_split_concat(self.df_baseline, self.df_current, test_size=0.25, random_state=42)
        self._df_concat = pd.concat([train, test]).reset_index(drop=True)

        indice_cat = cat_features_indices(train)

        feature_columns = [col for col in train.columns if col != "target"]

        train_pool_cat = catboost.Pool(
            data=train[feature_columns], label=train["target"].astype(int), cat_features=indice_cat
        )
        test_pool_cat = catboost.Pool(
            data=test[feature_columns], label=test["target"].astype(int), cat_features=indice_cat
        )
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

        datadrift_classifier = datadrift_classifier.fit(
            train_pool_cat,
            eval_set=test_pool_cat,
            silent=True,
            early_stopping_rounds=hyperparameter["early_stopping_rounds"],
        )

        train_logloss = datadrift_classifier.eval_metrics(train_pool_cat, "Logloss")
        best_iter_train = np.argmin(train_logloss["Logloss"]) + 1

        if best_iter_train < datadrift_classifier.tree_count_:
            datadrift_classifier.shrink(ntree_start=0, ntree_end=best_iter_train)

        self.xpl = SmartExplainer(
            label_dict={0: self.baseline_dataset_name, 1: self.current_dataset_name}, model=datadrift_classifier
        )

        x_test = test[feature_columns]
        y_test = test["target"]

        self.xpl.compile(x=x_test, y_target=y_test)
        self.xpl.compute_features_import(force=True)

        self.xpl.define_style(colors_dict=self.colors_dict)
        self.datadrift_classifier = datadrift_classifier
        if self.deployed_model:
            self.df_predict = self._predict(deployed_model=self.deployed_model, encoding=self.encoding)
        self.auc = roc_auc_score(y_test, datadrift_classifier.predict_proba(x_test)[:, 1])
        if self.deployed_model:
            self.feature_importance = self._compute_feature_importance(
                deployed_model=self.deployed_model, attr_importance=attr_importance
            )

        if self.deployed_model is not None:
            self.js_divergence = compute_js_divergence(
                self.df_predict.loc[lambda df: df["dataset"] == self.baseline_dataset_name, :]["Score"].values,
                self.df_predict.loc[lambda df: df["dataset"] == self.current_dataset_name, :]["Score"].values,
                n_bins=20,
            )
        if datadrift_file is not None:
            self.historical_auc = self._histo_datadrift_metric(
                datadrift_file=datadrift_file,
                date_compile_auc=date_compile_auc,
            )

        if self.deployed_model is not None:
            self.datadrift_stat_test = self._compute_datadrift_stat_test()

    def generate_report(
        self,
        output_file: str,
        project_info_file: str | None = None,
        title_story: str = "Drift Report",
        title_description: str = "",
        working_dir: str | None = None,
    ):
        """This method will generate an HTML report containing different information about the project.
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
            Working directory in which will be generated the notebook used to create the report and
            where the objects used to execute it will
            be saved. This parameter can be usefull if one wants to create its own custom report and
            debug the notebook used to generate the html report. If None, a temporary directory will be used

        Examples
        --------
        >>> SD.generate_report(
                output_file='report.html',
                project_info_file='project_info.yml',
                title_story="Drift project report",
                title_description="This document is a drift report of the score in production"
            )

        """
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
                modalities_analysis=self._modalities_analysis,
            )
        finally:
            if rm_working_dir:
                shutil.rmtree(working_dir)

    def _predict(self, deployed_model: Any, encoding: Any = None) -> pd.DataFrame:
        """Create an attributes df_predict with the computed score on both datasets

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
                ) from error
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
                ) from error
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
                ) from error
        return pd.concat(
            [
                df_baseline_pred.assign(dataset=self.baseline_dataset_name),
                df_current_pred.assign(dataset=self.current_dataset_name),
            ]
        ).reset_index(drop=True)

    def _compute_feature_importance(
        self, deployed_model: Any, attr_importance: str = "feature_importances_"
    ) -> pd.DataFrame:
        """Create an attributes feature_importance with the computed score on both datasets

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
        try:
            array_importance = getattr(deployed_model, attr_importance)
        except BaseException as error:
            raise Exception(
                """
            deployed_model used can't allow to get features importance on df_baseline
            - Error :
                            """
                + str(error)
            ) from error

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

    def _sampling(self, sampling: bool, sample_size: int, dataset: pd.DataFrame):
        """Return a sampling from the original dataframe

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

    def _histo_datadrift_metric(self, datadrift_file: str, date_compile_auc: date | None = None):
        """Method which computes datadrift metrics (AUC, and Jensen Shannon prediction divergence if the deployed_model
        is filled in) and append it into a dataframe that will be exported during the generate_report method

        Parameters
        ----------
        datadrift_file : str, (optional)
        date_compile_auc: str (optional)
            format dd/mm/yyyy use for specify date of compute drift, useful when compute few time drift
            for different time at the same moment

        Returns
        -------
        pandas.DataFrame or None
        Dataframe with dates, AUC and Jensen Shannon prediction divergence computed at this date

        """
        if date_compile_auc is None:
            date_compile_auc = date.today()
        s_date_compile_auc = date_compile_auc.strftime("%Y-%m-%d")
        print(f"The computed AUC on the X_test used to build datadrift_classifier is equal to: {self.auc}")

        df_auc = (
            pd.DataFrame({"date": [s_date_compile_auc], "auc": [self.auc], "JS_predict": [self.js_divergence]})
            if self.deployed_model is not None
            else pd.DataFrame({"date": [s_date_compile_auc], "auc": [self.auc]})
        )

        if datadrift_file is not None:
            if Path(datadrift_file).is_file() and datadrift_file.endswith(".csv"):
                histo_auc = pd.read_csv(datadrift_file).reset_index(drop=True)
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
                print(f"{datadrift_file} did not exist and was created. ")

            try:
                df_auc.to_csv(datadrift_file)
            except OSError as error:
                raise OSError("Can't save to csv the AUC metrics, error : " + str(error)) from error
        return df_auc

    def add_data_modeldrift(
        self,
        dataset: pd.DataFrame,
        metric: str = "performance",
        reference_columns: list[str] | None = None,
        year_col: str = "annee",
        month_col: str = "mois",
    ):
        """When method drift is specified, It will display in the report
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
        if reference_columns is None:
            reference_columns = []
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
            ) from error

    def _compute_datadrift_stat_test(self, max_size: int = 50000, categ_max: int = 20):
        """Calculates all statistical tests to analyze the drift of each feature

        Parameters
        ----------
        max_size : int
            Sets the maximum number of rows. If the datasets are larger there is sampling
        categ_max: int
            Maximum number of values per feature to apply the chi square test

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
            except BaseException as error:
                raise Exception(
                    f"""
                There is a problem with the format of {str(features)} column between the two datasets.
                Error:
                """
                    + str(error)
                ) from error
            test_results[features] = test

        return pd.DataFrame.from_dict(test_results, orient="index")

    def define_style(self, palette_name: str | None = None, colors_dict: dict | None = None):
        """The define_style function is a function that uses a palette or a dict
        to define the different styles used in the different outputs of Eurybia

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

    def save(self, path: str):
        """Save method allows user to save SmartDrift object on disk
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
            elif isinstance(getattr(self, att), DatasetAnalysis):
                da_dict = {}
                for att_da in self.da.__dict__.keys():
                    if isinstance(
                        getattr(self.da, att_da), (list, dict, pd.DataFrame, pd.Series, type(None), bool)
                    ) or att_da in ["model", "preprocessing", "postprocessing"]:
                        da_dict.update({att_da: getattr(self.da, att_da)})
                dict_to_save.update({att: da_dict})
        save_pickle(dict_to_save, path)

    @property
    def df_current(self) -> pd.DataFrame:
        """Getter"""
        return self._df_current

    @df_current.setter
    def df_current(self, val: pd.DataFrame) -> None:
        """Setter"""
        if not isinstance(val, pd.DataFrame):
            raise ValueError("df_current must be a pandas DataFrame")
        self._df_current = val

    @property
    def df_baseline(self) -> pd.DataFrame:
        """Getter"""
        return self._df_baseline

    @df_baseline.setter
    def df_baseline(self, val: pd.DataFrame) -> None:
        """Setter"""
        if not isinstance(val, pd.DataFrame):
            raise ValueError("df_baseline must be a pandas DataFrame")
        self._df_baseline = val

    @property
    def xpl(self) -> SmartExplainer:
        """Getter"""
        if not hasattr(self, "_xpl"):
            raise RuntimeError("SmartExplainer has not been initialized yet.")
        return self._xpl

    @xpl.setter
    def xpl(self, val: SmartExplainer) -> None:
        """Setter"""
        if not isinstance(val, SmartExplainer):
            raise ValueError("xpl must be a SmartExplainer instance.")
        self._xpl = val

    @property
    def df_predict(self) -> pd.DataFrame:
        """Getter"""
        if not hasattr(self, "_df_predict"):
            raise RuntimeError("df_predict has not been initialized yet.")
        return self._df_predict

    @df_predict.setter
    def df_predict(self, val: pd.DataFrame) -> None:
        """Setter"""
        if not isinstance(val, pd.DataFrame):
            raise ValueError("df_predict must be a pandas DataFrame.")
        self._df_predict = val

    @property
    def feature_importance(self) -> pd.DataFrame:
        """Getter"""
        if not hasattr(self, "_feature_importance"):
            raise RuntimeError("feature_importance has not been initialized yet.")
        return self._feature_importance

    @feature_importance.setter
    def feature_importance(self, val: pd.DataFrame) -> None:
        """Setter"""
        if not isinstance(val, pd.DataFrame):
            raise ValueError("feature_importance must be a pandas DataFrame.")
        self._feature_importance = val

    # @property
    # def pb_cols(self) -> dict[str, list[str]]:
    #     """Getter"""
    #     return self._pb_cols

    # @pb_cols.setter
    # def pb_cols(self, val: dict[str, list[str]]) -> None:
    #     """Setter"""
    #     if not isinstance(val, dict):
    #         raise ValueError("pb_cols must be a dictionary.")
    #     self._pb_cols = val

    # @property
    # def err_mods(self) -> dict[str, dict]:
    #     """Getter"""
    #     return self._err_mods

    # @err_mods.setter
    # def err_mods(self, val: dict[str, dict]) -> None:
    #     """Setter"""
    #     if not isinstance(val, dict):
    #         raise ValueError("err_mods must be a dictionary.")
    #     self._err_mods = val

    @property
    def auc(self) -> float:
        """Getter"""
        if not hasattr(self, "_auc"):
            raise RuntimeError("auc has not been initialized yet.")
        return self._auc

    @auc.setter
    def auc(self, val: float) -> None:
        """Setter"""
        if not isinstance(val, (float, int)):
            raise ValueError("auc must be of type float or int.")
        self._auc = float(val)

    @property
    def js_divergence(self) -> float:
        """Getter"""
        if not hasattr(self, "_js_divergence"):
            raise RuntimeError("js_divergence has not been initialized yet.")
        return self._js_divergence

    @js_divergence.setter
    def js_divergence(self, val: float) -> None:
        """Setter"""
        if not isinstance(val, (float, int)):
            raise ValueError("js_divergence must be of type float or int.")
        self._js_divergence = float(val)

    @property
    def historical_auc(self) -> pd.DataFrame | None:
        """Getter"""
        if not hasattr(self, "_historical_auc"):
            # raise RuntimeError("historical_auc has not been initialized yet.")
            return None
        return self._historical_auc

    @historical_auc.setter
    def historical_auc(self, val: pd.DataFrame) -> None:
        """Setter"""
        if not isinstance(val, pd.DataFrame):
            raise ValueError("historical_auc must be a pandas DataFrame.")
        self._historical_auc = val

    @property
    def data_modeldrift(self) -> pd.DataFrame | None:
        """Getter"""
        if not hasattr(self, "_data_modeldrift"):
            # raise RuntimeError("data_modeldrift has not been initialized yet.")
            return None
        return self._data_modeldrift

    @data_modeldrift.setter
    def data_modeldrift(self, val: pd.DataFrame) -> None:
        """Setter"""
        if not isinstance(val, pd.DataFrame):
            raise ValueError("data_modeldrift must be a pandas DataFrame.")
        self._data_modeldrift = val

    @property
    def current_dataset_name(self) -> str:
        """Returns the display name of df_current"""
        return self._dataset_names[0]

    @property
    def baseline_dataset_name(self) -> str:
        """Returns the display name of df_baseline"""
        return self._dataset_names[1]

    @property
    def encoding(self) -> Any:
        """Getter"""
        return self._encoding

    @encoding.setter
    def encoding(self, val: Any) -> None:
        """Setter"""
        self._encoding = val

    @property
    def deployed_model(self) -> Any:
        """Getter"""
        return self._deployed_model

    @deployed_model.setter
    def deployed_model(self, val: Any) -> None:
        """Setter"""
        self._deployed_model = val

    @property
    def ignore_cols(self) -> list[str]:
        """Getter"""
        return self._ignore_cols

    @ignore_cols.setter
    def ignore_cols(self, val: list[str]) -> None:
        """Setter"""
        if not isinstance(val, list):
            raise ValueError("ignore_cols must be a list.")
        self._ignore_cols = val

    @property
    def datadrift_stat_test(self) -> pd.DataFrame:
        """Getter"""
        if not hasattr(self, "_datadrift_stat_test"):
            raise RuntimeError("datadrift_stat_test has not been initialized yet.")
        return self._datadrift_stat_test

    @datadrift_stat_test.setter
    def datadrift_stat_test(self, val: pd.DataFrame) -> None:
        """Setter"""
        if not isinstance(val, pd.DataFrame):
            raise ValueError("datadrift_stat_test must be a pandas DataFrame.")
        self._datadrift_stat_test = val

    @property
    def df_concat(self) -> pd.DataFrame:
        """Getter"""
        if not hasattr(self, "_df_concat"):
            raise RuntimeError("df_concat has not been initialized yet.")
        return self._df_concat

    @df_concat.setter
    def df_concat(self, val: pd.DataFrame | None) -> None:
        """Setter"""
        if val is not None and not isinstance(val, pd.DataFrame):
            raise ValueError("df_concat must be a pandas DataFrame or None.")
        self._df_concat = val

    @property
    def datadrift_target(self) -> str:
        """Getter"""
        return self._datadrift_target

    @property
    def plot(self) -> SmartPlotter:
        """Getter"""
        return self._plot

    @property
    def da(self) -> DatasetAnalysis:
        """Getter"""
        if not hasattr(self, "_da"):
            raise RuntimeError("da has not been initialized yet.")
        return self._da

    @da.setter
    def da(self, val: DatasetAnalysis) -> None:
        """Setter"""
        if not isinstance(val, DatasetAnalysis):
            raise ValueError(f"da must be a of type {DatasetAnalysis.__name__}.")
        self._da = val
