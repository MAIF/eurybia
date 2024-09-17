"""
Smart plotter module
"""

import copy

# ----- Eurybia packages
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from eurybia.report.common import VarType, compute_col_types
from eurybia.style.style_utils import colors_loading, define_style, select_palette


class SmartPlotter:
    """
    The smartplotter class includes all the methods used to display graphics

    Each SmartPlotter method is easy to use from a Smart Drift object,
    just use the following syntax

    Attributes
    ----------
    smartdrift: object
        SmartDrift object
    _palette_name : str (default: 'eurybia')
        Name of the palette used for the colors of the report (refer to style folder).
    _style_dict: dict
            Dict contains dicts of the colors used in the different plots
    Example
    --------
    >>> SD = Smartdrift()
    >>> SD.compile()
    >>> SD.plot.my_plot_method(param=value)

    """

    def __init__(self, smartdrift):
        self._palette_name = list(colors_loading().keys())[0]
        self._style_dict = define_style(select_palette(colors_loading(), self._palette_name))
        self.smartdrift = smartdrift

    def generate_fig_univariate(
        self,
        col: str,
        hue: Optional[str] = None,
        df_all: Optional[pd.DataFrame] = None,
        dict_color_palette: Optional[dict] = None,
    ) -> plt.Figure:
        """
        Returns a plotly figure containing the distribution of any kind of feature
        (continuous, categorical).

        If the feature is categorical and contains too many categories, the smallest
        categories are grouped into a new 'Other' category so that the graph remains
        readable.

        The input dataframe should contain the column of interest and a column that is used
        to distinguish two types of values (ex. 'train' and 'test')

        Parameters
        ----------
        df_all : pd.DataFrame
            The input dataframe that contains the column of interest
        col : str
            The column of interest
        hue : str
            The column used to distinguish the values (ex. 'train' and 'test')
        type: str
            The type of the series ('continous' or 'categorical')

        Returns
        -------
        plotly.graph_objs._figure.Figure
        """
        if hue is None:
            hue = self.smartdrift._datadrift_target
        if df_all is None:
            df_all = self.smartdrift._df_concat
            df_all.loc[df_all[hue] == 0, hue] = list(self.smartdrift.dataset_names.keys())[1]
            df_all.loc[df_all[hue] == 1, hue] = list(self.smartdrift.dataset_names.keys())[0]
        if dict_color_palette is None:
            dict_color_palette = self._style_dict
        col_types = compute_col_types(df_all=df_all)
        if col_types[col] == VarType.TYPE_NUM:
            fig = self.generate_fig_univariate_continuous(df_all, col, hue=hue, dict_color_palette=dict_color_palette)
        elif col_types[col] == VarType.TYPE_CAT:
            fig = self.generate_fig_univariate_categorical(df_all, col, hue=hue, dict_color_palette=dict_color_palette)
        else:
            raise NotImplementedError("Series dtype not supported")
        return fig

    def generate_fig_univariate_continuous(
        self,
        df_all: pd.DataFrame,
        col: str,
        hue: str,
        dict_color_palette: dict,
        template: Optional[str] = None,
        title: Optional[str] = None,
        xaxis_title: Optional[dict] = None,
        yaxis_title: Optional[dict] = None,
        xaxis: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        hovermode: Optional[str] = None,
    ) -> plotly.graph_objs._figure.Figure:
        """
        Returns a plotly figure containing the distribution of a continuous feature.

        Parameters
        ----------
        df_all : pd.DataFrame
            The input dataframe that contains the column of interest
        col : str
            The column of interest
        hue : str
            The column used to distinguish the values (ex. 'train' and 'test')
        template: str, , optional
            Template (background style) for the plot
        title: str, optional
            Plot title
        xaxis_title: str, , optional
            X axis title
        yaxis_title: str, , optional
            y axis title
        xaxis: str, , optional
            X axis options (spike line, margin, range ...)
        height: str, , optional
            Height of the plot
        width: str, , optional
            Width of the plot
        hovermode: str,n , optional
            Type of labels displaying on mouse hovering
        Returns
        -------
        plotly.graph_objs._figure.Figure
        """
        df_all[col] = df_all[col].fillna(0)
        datasets = [df_all[df_all[hue] == val][col].values.tolist() for val in df_all[hue].unique()]
        group_labels = [str(val) for val in df_all[hue].unique()]
        colors = list(self._style_dict["univariate_cont_bar"].values())
        if group_labels[0] == "Current dataset":
            group_labels = ["Baseline dataset", "Current dataset"]

        fig = ff.create_distplot(
            datasets,
            group_labels=group_labels,
            colors=list(colors),
            show_hist=False,
            show_curve=True,
            show_rug=False,
        )
        if template is None:
            template = self._style_dict["template"]
        if title is None:
            title = self._style_dict["dict_title"]
        if xaxis_title is None:
            xaxis_title = self._style_dict["dict_xaxis_continuous"]
            xaxis_title["text"] = col
        if yaxis_title is None:
            yaxis_title = self._style_dict["dict_yaxis_continuous"]
        if xaxis is None:
            xaxis = self._style_dict["dict_xaxis"]
        if height is None:
            height = self._style_dict["height"]
        if width is None:
            width = self._style_dict["width"]
        if hovermode is None:
            hovermode = self._style_dict["hovermode"]

        fig.update_layout(
            template=template,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            xaxis=xaxis,
            height=height,
            width=width,
            hovermode=hovermode,
        )
        fig.update_traces(hovertemplate="%{y:.2f}", showlegend=True)
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        return fig

    def generate_fig_univariate_categorical(
        self,
        df_all: pd.DataFrame,
        col: str,
        hue: str,
        dict_color_palette: dict,
        nb_cat_max: int = 15,
        template: Optional[str] = None,
        title: Optional[str] = None,
        xaxis_title: Optional[dict] = None,
        yaxis_title: Optional[dict] = None,
        xaxis: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        hovermode: Optional[str] = None,
        legend: Optional[str] = None,
    ) -> plotly.graph_objs._figure.Figure:
        """
        Returns a plotly figure containing the distribution of a categorical feature.

        If the feature is categorical and contains too many categories, the smallest
        categories are grouped into a new 'Other' category so that the graph remains
        readable.

        Parameters
        ----------
        df_all : pd.DataFrame
            The input dataframe that contains the column of interest
        col : str
            The column of interest
        hue : str
            The column used to distinguish the values (ex. 'train' and 'test')
        nb_cat_max : int
            The number max of categories to be displayed. If the number of categories
            is greater than nb_cat_max then groups smallest categories into a new
            'Other' category
        template: str, optional
            Template (background style) for the plot
        title: str, optional
            Plot title
        xaxis_title: str, optional
            X axis title
        yaxis_title: str, optional
            y axis title
        xaxis: str, optional
            X axis options (spike line, margin, range ...)
        height: str, optional
            Height of the plot
        width: str, optional
            Width of the plot
        hovermode: str, optional
            Type of labels displaying on mouse hovering
        legend: str, optional
            Axis legends
        Returns
        -------
        plotly.graph_objs._figure.Figure
        """
        df_cat = df_all.groupby([col, hue]).agg({col: "count"}).rename(columns={col: "count"}).reset_index()
        df_cat["Percent"] = df_cat["count"] * 100 / df_cat.groupby(hue)["count"].transform("sum")

        if pd.api.types.is_numeric_dtype(df_cat[col].dtype):
            df_cat = df_cat.sort_values(col, ascending=True)
            df_cat[col] = df_cat[col].astype(str)

        nb_cat = df_cat.groupby([col]).agg({"count": "sum"}).reset_index()[col].nunique()

        if nb_cat > nb_cat_max:
            df_cat = self._merge_small_categories(df_cat=df_cat, col=col, hue=hue, nb_cat_max=nb_cat_max)

        df_to_sort = df_cat.copy().reset_index(drop=True)
        df_to_sort["Sorted_indicator"] = df_to_sort.sort_values([col]).groupby([col])["Percent"].diff()
        df_to_sort["Sorted_indicator"] = np.abs(df_to_sort["Sorted_indicator"])
        df_sorted = df_to_sort.dropna()[[col, "Sorted_indicator"]]

        df_cat = (
            df_cat.merge(df_sorted, how="left", on=col)
            .sort_values("Sorted_indicator", ascending=True)
            .drop("Sorted_indicator", axis=1)
        )

        df_cat["Percent_displayed"] = df_cat["Percent"].apply(lambda row: str(round(row, 2)) + " %")

        modalities = df_cat[hue].unique().tolist()

        fig1 = px.bar(
            df_cat[df_cat[hue] == modalities[0]],
            x="Percent",
            y=col,
            orientation="h",
            barmode="group",
            color=hue,
            text="Percent_displayed",
        )
        fig1.update_traces(marker_color=list(self._style_dict["univariate_cat_bar"].values())[1], showlegend=True)

        fig2 = px.bar(
            df_cat[df_cat[hue] == modalities[1]],
            x="Percent",
            y=col,
            orientation="h",
            barmode="group",
            color=hue,
            text="Percent_displayed",
        )
        fig2.update_traces(marker_color=list(self._style_dict["univariate_cat_bar"].values())[0], showlegend=True)

        fig = fig1.add_trace(fig2.data[0])

        fig.update_xaxes(showgrid=False, showticklabels=True)
        fig.update_yaxes(showgrid=False, showticklabels=True, automargin=True)
        fig.update_traces(showlegend=True, textposition="outside", cliponaxis=False)

        if template is None:
            template = self._style_dict["template"]
        if title is None:
            title = self._style_dict["dict_title"]
        if xaxis_title is None:
            xaxis_title = self._style_dict["dict_xaxis_title"]
        if yaxis_title is None:
            yaxis_title = self._style_dict["dict_yaxis_title"]
            yaxis_title["text"] = col
        if height is None:
            height = self._style_dict["height"]
        if width is None:
            width = self._style_dict["width"]
        if hovermode is None:
            hovermode = self._style_dict["hovermode"]
        if legend is None:
            legend = self._style_dict["dict_legend"]

        fig.update_layout(
            template=template,
            title=title,
            xaxis_title=xaxis_title,
            height=height,
            width=width,
            yaxis_title=yaxis_title,
            hovermode=hovermode,
            legend=legend,
            xaxis_range=[0, max(df_cat["Percent"]) + 10],
        )

        return fig

    def _merge_small_categories(self, df_cat: pd.DataFrame, col: str, hue: str, nb_cat_max: int) -> pd.DataFrame:
        """
        Merges categories of column 'col' of df_cat into 'Other' category so that
        the number of categories is less than nb_cat_max.
        """
        df_cat_sum_hue = df_cat.groupby([col]).agg({"count": "sum"}).reset_index()
        list_cat_to_merge = df_cat_sum_hue.sort_values("count", ascending=False)[col].to_list()[nb_cat_max - 1 :]
        df_cat_other = (
            df_cat.loc[df_cat[col].isin(list_cat_to_merge)].groupby(hue, as_index=False)[["count", "Percent"]].sum()
        )
        df_cat_other[col] = "Other"
        return pd.concat([df_cat.loc[~df_cat[col].isin(list_cat_to_merge)], df_cat_other])

    def scatter_feature_importance(
        self, feature_importance: pd.DataFrame = None, datadrift_stat_test: pd.DataFrame = None
    ) -> plotly.graph_objs._figure.Figure:
        """
        Displays scatter of feature importance between drift
        model and production one extracted from a datasets created
        during the compile step.

        Parameters
        ----------
        feature_importance : pd.DataFrame, optional
            DataFrame containing feature importance for each features from production and drift model.
        datadrift_stat_test: pd.DataFrame, optional
            DataFrame containing the result of datadrift univariate tests
        Returns
        -------
        plotly.express.scatter
        """
        dict_t = copy.deepcopy(self._style_dict["dict_title"])
        dict_xaxis = copy.deepcopy(self._style_dict["dict_xaxis_title"])
        dict_yaxis = copy.deepcopy(self._style_dict["dict_yaxis_title"])
        title = "<b>Datadrift Vs Feature Importance</b>"
        dict_t["text"] = title
        dict_xaxis["text"] = "Datadrift Importance"
        dict_yaxis["text"] = "Feature Importance - Deployed Model"

        if feature_importance is None:
            feature_importance = self.smartdrift.feature_importance.set_index("feature")
        if datadrift_stat_test is None:
            datadrift_stat_test = self.smartdrift.datadrift_stat_test

        data = datadrift_stat_test.join(feature_importance)
        data["features"] = data.index
        # symbols
        stat_test_list = list(data["testname"].unique())
        symbol_list = [0, 13]
        symbol_dict = dict(zip(stat_test_list, symbol_list))

        hv_text = [
            f"<b>Feature: {feat}</b><br />Deployed Model Importance: {depimp*100:.1f}%<br />"
            + f"Datadrift test: {t} - pvalue: {pv:.5f}<br />"
            + f"Datadrift model Importance: {ddrimp*100:.1f}"
            for feat, depimp, t, pv, ddrimp in zip(
                *map(data.get, ["features", "deployed_model", "testname", "pvalue", "datadrift_classifier"])
            )
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["datadrift_classifier"],
                y=data["deployed_model"],
                marker_symbol=datadrift_stat_test["testname"].apply(lambda x: symbol_dict[x]),
                mode="markers",
                showlegend=False,
                hovertext=hv_text,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

        fig.update_traces(marker={"size": 15, "opacity": 0.8, "line": {"width": 0.8, "color": "white"}})

        fig.data[0].marker.color = data["pvalue"]
        fig.data[0].marker.coloraxis = "coloraxis"
        fig.layout.coloraxis.colorscale = self._style_dict["featimportance_colorscale"]
        fig.layout.coloraxis.colorbar = {"title": {"text": "Univariate<br />DataDrift Test<br />Pvalue"}}

        height = self._style_dict["height"]
        width = self._style_dict["width"]
        hovermode = self._style_dict["hovermode"]
        template = self._style_dict["template"]

        fig.update_layout(
            template=template,
            title=dict_t,
            xaxis_title=dict_xaxis,
            yaxis_title=dict_yaxis,
            height=height,
            width=width,
            hovermode=hovermode,
        )

        return fig

    def generate_historical_datadrift_metric(
        self,
        datadrift_historical: pd.DataFrame = None,
        template: Optional[str] = None,
        title: Optional[str] = None,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[str] = None,
        xaxis: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        hovermode: Optional[str] = None,
    ) -> plotly.graph_objs._figure.Figure:
        """
        Displays line plot of the evolution of the datadrift metrics :
        AUC of Datadrift classifier and if deployed_model fill, Jensen Shannon divergence of distribution of prediction
        Parameters
        ----------
        datadrift_historical : pd.DataFrame
            DataFrame with date, datadrif classifer auc and jensen shannon prediction divergence if deployed_model fill
        template: str, optional
            Template (background style) for the plot
        title: str, optional
            Plot title
        xaxis_title: str, optional
            X axis title
        yaxis_title: str, optional
            y axis title
        xaxis: str, optional
            X axis options (spike line, margin, range ...)
        height: str, optional
            Height of the plot
        width: str, optional
            Width of the plot
        hovermode: str, optional
            Type of labels displaying on mouse hovering
        Returns
        -------
        plotly.express.line
        """
        if datadrift_historical is None:
            datadrift_historical = self.smartdrift.historical_auc
        if datadrift_historical is not None:
            if self.smartdrift.deployed_model is not None:
                datadrift_historical = datadrift_historical[["date", "auc", "JS_predict"]]
                datadrift_historical = (
                    datadrift_historical.groupby(["date"])[["auc", "JS_predict"]].mean().reset_index()
                )
                datadrift_historical.sort_values(by="date", inplace=True)
            else:
                datadrift_historical = datadrift_historical[["date", "auc"]]
                datadrift_historical = datadrift_historical.groupby("date")["auc"].mean().reset_index()
                datadrift_historical.sort_values(by="date", inplace=True)

            datadrift_historical["auc_displayed"] = datadrift_historical["auc"].round(2)

            if self.smartdrift.deployed_model is not None:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=datadrift_historical["date"], y=datadrift_historical["auc"], name="Datadrift classifier AUC"
                    ),
                    secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(
                        x=datadrift_historical["date"],
                        y=datadrift_historical["JS_predict"],
                        name="Jensen_Shannon Prediction Divergence",
                    ),
                    secondary_y=True,
                )

                fig.update_layout(title_text="Evolution of data drift")
                fig.update_yaxes(title_text="<b>Datadrift classifier AUC</b>  ", secondary_y=False)
                fig.update_yaxes(title_text="<b>Jensen_Shannon Prediction Divergence</b> ", secondary_y=True)
                fig.update_yaxes(range=[0.5, 1], secondary_y=False)
                fig.update_yaxes(range=[0, 0.3], secondary_y=True)
            else:
                fig = px.line(
                    datadrift_historical,
                    x="date",
                    y="auc",
                    title="AUC's Evolution of Datadrift classifier",
                    text="auc_displayed",
                )
                fig.update_yaxes(title_text="<b>Datadrift classifier AUC</b>")
                fig.update_yaxes(range=[0.5, 1])

            fig.update_traces(textposition="bottom right")

            if template is None:
                template = self._style_dict["template"]
            if title is None:
                title = self._style_dict["dict_title"]
            if xaxis_title is None:
                xaxis_title = self._style_dict["dict_xaxis_title"]
            if height is None:
                height = self._style_dict["height"]
            if width is None:
                width = self._style_dict["width"]
            if hovermode is None:
                hovermode = self._style_dict["hovermode"]

            fig.update_xaxes(showgrid=False)
            fig.update_layout(
                template=template,
                title=title,
                xaxis_title=xaxis_title,
                height=height,
                width=width,
                hovermode=hovermode,
            )
            return fig

    def generate_modeldrift_data(
        self,
        data_modeldrift: pd.DataFrame = None,
        metric: str = "performance",
        reference_columns: list = list(),
        template: Optional[str] = None,
        title: Optional[str] = None,
        xaxis_title: Optional[str] = None,
        yaxis_title: Optional[dict] = None,
        xaxis: Optional[str] = None,
        height: Optional[str] = None,
        width: Optional[str] = None,
        hovermode: Optional[str] = None,
    ) -> plotly.graph_objs._figure.Figure:
        """
        Displays line plot of the evolution of the Lift computed for deployed model with several criterias.

        Parameters
        ----------
        data_modeldrift : pd.DataFrame
            DataFrame containing the aggregated informations to display modeldrift.
        metric : str
            Column name of the metric computed
        reference_columns : list
            list of reference columns used to display the metric according to different criteria
        title: str, optional
            Plot title
        xaxis_title: str, optional
            X axis title
        yaxis_title: dict, optional
            y axis title
        xaxis: str, optional
            X axis options (spike line, margin, range ...)
        height: str, optional
            Height of the plot
        width: str, optional
            Width of the plot
        hovermode: str, optional
            Type of labels displaying on mouse hovering
        Returns
        -------
        plotly.express.line
        """
        if data_modeldrift is None:
            data_modeldrift = self.smartdrift.data_modeldrift
            if data_modeldrift is None:
                raise ValueError(
                    """You should run the add_data_modeldrift method before displaying model drift performances.
                For more information see the documentation"""
                )
        data_modeldrift[metric] = data_modeldrift[metric].apply(
            lambda row: round(row, len([char for char in str(row).split(".")[1] if char == "0"]) + 3)
        )

        fig = px.line(
            data_modeldrift,
            x="Date",
            y=metric,
            hover_data=reference_columns,
            title="Performance's Evolution on deployed model",
            text=metric,
        )

        fig.update_traces(textposition="top right")

        if template is None:
            template = self._style_dict["template"]
        if title is None:
            title = self._style_dict["dict_title"]
        if xaxis_title is None:
            xaxis_title = self._style_dict["dict_xaxis_title"]
        if yaxis_title is None:
            yaxis_title = self._style_dict["dict_yaxis_title"]
            yaxis_title["text"] = metric
        if height is None:
            height = self._style_dict["height"]
        if width is None:
            width = self._style_dict["width"]
        if hovermode is None:
            hovermode = self._style_dict["hovermode"]

        fig.update_xaxes(showgrid=False)
        fig.update_layout(
            template=template,
            title=title,
            xaxis_title=xaxis_title,
            height=height,
            width=width,
            yaxis_title=yaxis_title,
            hovermode=hovermode,
        )

        fig.data[0].line.color = self._style_dict["datadrift_historical"]
        fig.data[-1].marker.color = self._style_dict["datadrift_historical"]

        return fig

    def define_style_attributes(self, colors_dict):
        """
        define_style_attributes allows Eurybia user to change the color of plot
        Parameters
        ----------
        colors_dict: dict
            Dict of the colors used in the different plots
        """
        self._style_dict = define_style(colors_dict)

        if hasattr(self, "pred_colorscale"):
            delattr(self, "pred_colorscale")

    def generate_indicator(
        self,
        fig_value: float,
        min_gauge: float = 0.5,
        max_gauge: float = 1,
        height: Optional[float] = 300,
        width: Optional[float] = 500,
        title: Optional[str] = "Metric",
    ) -> plotly.graph_objs._figure.Figure:
        """
        Displays an indicator in a colorbar
        Parameters
        ----------
        fig_value: float
            Value to display on figure
        min_gauge: float, (default: 0.5)
            range min in gauge display
        max_gauge: float, (default: 1)
            range max in gauge display
        height: str, optional
            Height of the plot
        width: str, optional
            Width of the plot
        title: str, optional
            Plot title
        """
        color = sns.blend_palette(["green", "yellow", "orange", "red"], 100)
        color = color.as_hex()
        list_color_glob = list()
        threshold = [i for i in np.arange(min_gauge, max_gauge, (max_gauge - min_gauge) / len(color))]
        for i in range(1, len(threshold) + 1):
            dict_color = dict()
            if i == len(threshold):
                rang = [threshold[i - 1], 1]
            else:
                rang = [threshold[i - 1], threshold[i]]
            dict_color["range"] = rang
            dict_color["color"] = color[i - 1]
            list_color_glob.append(dict_color)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=round(fig_value, 2),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": title, "align": "center", "font": {"size": 20}},
                gauge={
                    "axis": {"range": [min_gauge, max_gauge], "ticktext": ["No Drift", "High Drift"], "tickwidth": 1},
                    "bar": {"color": "black"},
                    "borderwidth": 0,
                    "steps": list_color_glob,
                },
            )
        )
        fig.update_layout(
            height=height,
            width=width,
        )

        return fig
