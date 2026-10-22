"""Smart plotter module"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eurybia import SmartDrift

import copy
from html import escape
from math import cos, pi, sin

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
from plotly import graph_objs as go
from plotly.subplots import make_subplots

from eurybia.report.common import VarType, compute_col_types
from eurybia.style.style_utils import colors_loading, define_style, select_palette


class SmartPlotter:
    """The smartplotter class includes all the methods used to display graphics

    Each SmartPlotter method is easy to use from a Smart Drift object,
    just use the following syntax

    Attributes:
    ----------
    smartdrift: object
        SmartDrift object
    style_dict: dict
            Dict contains dicts of the colors used in the different plots

    Example:
    -------
    >>> SD = Smartdrift()
    >>> SD.compile()
    >>> SD.plot.my_plot_method(param=value)

    """

    def __init__(self, smartdrift: SmartDrift):
        """Initialize a SmartPlotter.

        Args:
            smartdrift (SmartDrift): a SmartDrift instance

        """
        palette_name = list(colors_loading().keys())[0]
        self._style_dict = define_style(select_palette(colors_loading(), palette_name))
        self._smartdrift = smartdrift

    @property
    def style_dict(self) -> dict:
        """Getter"""
        return self._style_dict

    @style_dict.setter
    def style_dict(self, val: dict) -> None:
        """Setter"""
        if not isinstance(val, dict):
            raise ValueError("style_dict must be a dictionary.")
        self._style_dict = val

    @property
    def smartdrift(self) -> SmartDrift:
        """Getter"""
        return self._smartdrift

    @smartdrift.setter
    def smartdrift(self, val: SmartDrift) -> None:
        """Setter"""
        if not isinstance(val, SmartDrift):
            raise ValueError("style_dict must be of type SmartDrift.")
        self._smartdrift = val

    def generate_fig_univariate(
        self,
        col: str,
        hue: str | None = None,
        df_all: pd.DataFrame | None = None,
        dict_color_palette: dict | None = None,
    ) -> go.Figure:
        """Returns a plotly figure containing the distribution of any kind of feature
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
            hue = self.smartdrift.datadrift_target
        if df_all is None:
            df_all = self.smartdrift.df_concat
            df_all[hue] = df_all[hue].astype("object")
            df_all.loc[df_all[hue] == 0, hue] = self.smartdrift.current_dataset_name
            df_all.loc[df_all[hue] == 1, hue] = self.smartdrift.baseline_dataset_name
        if dict_color_palette is None:
            dict_color_palette = self.style_dict
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
        dict_color_palette: dict,  # FIXME: unused
        template: str | None = None,
        title: str | None = None,
        xaxis_title: dict | None = None,
        yaxis_title: dict | None = None,
        xaxis: str | None = None,
        height: str | None = None,
        width: str | None = None,
        hovermode: str | None = None,
    ) -> plotly.graph_objs._figure.Figure:
        """Returns a plotly figure containing the distribution of a continuous feature.

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
        colors = list(self.style_dict["univariate_cont_bar"].values())
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
            template = self.style_dict["template"]
        if title is None:
            title = self.style_dict["dict_title"]
        if xaxis_title is None:
            xaxis_title = self.style_dict["dict_xaxis_continuous"]
            xaxis_title["text"] = col
        if yaxis_title is None:
            yaxis_title = self.style_dict["dict_yaxis_continuous"]
        if xaxis is None:
            xaxis = self.style_dict["dict_xaxis"]
        if height is None:
            height = self.style_dict["height"]
        if width is None:
            width = self.style_dict["width"]
        if hovermode is None:
            hovermode = self.style_dict["hovermode"]

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
        dict_color_palette: dict,  # FIXME: unused
        nb_cat_max: int = 15,
        template: str | None = None,
        title: str | None = None,
        xaxis_title: dict | None = None,
        yaxis_title: dict | None = None,
        xaxis: str | None = None,  # FIXME: unused
        height: str | None = None,
        width: str | None = None,
        hovermode: str | None = None,
        legend: str | None = None,
    ) -> plotly.graph_objs._figure.Figure:
        """Returns a plotly figure containing the distribution of a categorical feature.

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
        fig1.update_traces(marker_color=list(self.style_dict["univariate_cat_bar"].values())[1], showlegend=True)

        fig2 = px.bar(
            df_cat[df_cat[hue] == modalities[1]],
            x="Percent",
            y=col,
            orientation="h",
            barmode="group",
            color=hue,
            text="Percent_displayed",
        )
        fig2.update_traces(marker_color=list(self.style_dict["univariate_cat_bar"].values())[0], showlegend=True)

        fig = fig1.add_trace(fig2.data[0])

        fig.update_xaxes(showgrid=False, showticklabels=True)
        fig.update_yaxes(showgrid=False, showticklabels=True, automargin=True)
        fig.update_traces(showlegend=True, textposition="outside", cliponaxis=False)

        if template is None:
            template = self.style_dict["template"]
        if title is None:
            title = self.style_dict["dict_title"]
        if xaxis_title is None:
            xaxis_title = self.style_dict["dict_xaxis_title"]
        if yaxis_title is None:
            yaxis_title = self.style_dict["dict_yaxis_title"]
            yaxis_title["text"] = col
        if height is None:
            height = self.style_dict["height"]
        if width is None:
            width = self.style_dict["width"]
        if hovermode is None:
            hovermode = self.style_dict["hovermode"]
        if legend is None:
            legend = self.style_dict["dict_legend"]

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
        """Merges categories of column 'col' of df_cat into 'Other' category so that
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
        self, feature_importance: pd.DataFrame | None = None, datadrift_stat_test: pd.DataFrame | None = None
    ) -> plotly.graph_objs._figure.Figure:
        """Displays scatter of feature importance between drift
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
        dict_t = copy.deepcopy(self.style_dict["dict_title"])
        dict_xaxis = copy.deepcopy(self.style_dict["dict_xaxis_title"])
        dict_yaxis = copy.deepcopy(self.style_dict["dict_yaxis_title"])
        title = "<b>Datadrift Vs Feature Importance</b>"
        dict_t["text"] = title
        dict_xaxis["text"] = "Datadrift Importance"
        dict_yaxis["text"] = "Feature Importance - Deployed Model"

        if feature_importance is None:
            if self.smartdrift.feature_importance is None:
                raise RuntimeError("SmartDrift().feature_importance should not be None at this point.")
            feature_importance = self.smartdrift.feature_importance.set_index("feature")
        if datadrift_stat_test is None:
            if self.smartdrift.datadrift_stat_test is None:
                raise RuntimeError("SmartDrift().datadrift_stat_test should not be None at this point.")
            datadrift_stat_test = self.smartdrift.datadrift_stat_test

        data = datadrift_stat_test.join(feature_importance)
        data["features"] = data.index
        # symbols
        stat_test_list = list(data["testname"].unique())
        symbol_list = [0, 13]
        symbol_dict = dict(zip(stat_test_list, symbol_list, strict=True))

        hv_text = [
            f"<b>Feature: {feat}</b><br />Deployed Model Importance: {depimp * 100:.1f}%<br />"
            + f"Datadrift test: {t} - pvalue: {pv:.5f}<br />"
            + f"Datadrift model Importance: {ddrimp * 100:.1f}"
            for feat, depimp, t, pv, ddrimp in zip(
                *map(data.get, ["features", "deployed_model", "testname", "pvalue", "datadrift_classifier"]),
                strict=True,
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
        fig.layout.coloraxis.colorscale = self.style_dict["featimportance_colorscale"]
        fig.layout.coloraxis.colorbar = {"title": {"text": "Univariate<br />DataDrift Test<br />Pvalue"}}

        height = self.style_dict["height"]
        width = self.style_dict["width"]
        hovermode = self.style_dict["hovermode"]
        template = self.style_dict["template"]

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
        template: str | None = None,
        title: str | None = None,
        xaxis_title: str | None = None,
        yaxis_title: str | None = None,  # FIXME: unused
        xaxis: str | None = None,  # FIXME: unused
        height: str | None = None,
        width: str | None = None,
        hovermode: str | None = None,
    ) -> plotly.graph_objs._figure.Figure:
        """Displays line plot of the evolution of the datadrift metrics :
        AUC of the datadrift classifier and, if deployed_model is set, the Jensen-Shannon divergence between predictions

        Parameters
        ----------
        datadrift_historical : pd.DataFrame
            DataFrame with dates, drift-classifier AUC and Jensen-Shannon prediction divergence
            when deployed_model is set.
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

            datadrift_historical["auc_for_drift"] = np.maximum(
                datadrift_historical["auc"], 1 - datadrift_historical["auc"]
            )
            datadrift_historical["auc_displayed"] = datadrift_historical["auc_for_drift"].round(2)

            if self.smartdrift.deployed_model is not None:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=datadrift_historical["date"],
                        y=datadrift_historical["auc_for_drift"],
                        name="Datadrift classifier AUC",
                    ),
                    secondary_y=False,
                )

                fig.add_trace(
                    go.Scatter(
                        x=datadrift_historical["date"],
                        y=datadrift_historical["JS_predict"],
                        name="Jensen-Shannon Prediction Divergence",
                    ),
                    secondary_y=True,
                )

                fig.update_layout(title_text="Evolution of data drift")
                fig.update_yaxes(title_text="<b>Datadrift classifier AUC</b>  ", secondary_y=False)
                fig.update_yaxes(title_text="<b>Jensen-Shannon Prediction Divergence</b> ", secondary_y=True)
                fig.update_yaxes(range=[0.5, 1], secondary_y=False)
                fig.update_yaxes(range=[0, 1], secondary_y=True)
            else:
                fig = px.line(
                    datadrift_historical,
                    x="date",
                    y="auc_for_drift",
                    title="AUC's Evolution of Datadrift classifier",
                    text="auc_displayed",
                )
                fig.update_yaxes(title_text="<b>Datadrift classifier AUC</b>")
                fig.update_yaxes(range=[0.5, 1])

            fig.update_traces(textposition="bottom right")

            if template is None:
                template = self.style_dict["template"]
            if title is None:
                title = self.style_dict["dict_title"]
            if xaxis_title is None:
                xaxis_title = self.style_dict["dict_xaxis_title"]
            if height is None:
                height = self.style_dict["height"]
            if width is None:
                width = self.style_dict["width"]
            if hovermode is None:
                hovermode = self.style_dict["hovermode"]

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
        reference_columns: list | None = None,
        template: str | None = None,
        title: str | None = None,
        xaxis_title: str | None = None,
        yaxis_title: dict | None = None,
        xaxis: str | None = None,  # FIXME: unused
        height: str | None = None,
        width: str | None = None,
        hovermode: str | None = None,
    ) -> plotly.graph_objs._figure.Figure:
        """Displays line plot of the evolution of the Lift computed for deployed model with several criterias.

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
        if reference_columns is None:
            reference_columns = list()

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
            template = self.style_dict["template"]
        if title is None:
            title = self.style_dict["dict_title"]
        if xaxis_title is None:
            xaxis_title = self.style_dict["dict_xaxis_title"]
        if yaxis_title is None:
            yaxis_title = self.style_dict["dict_yaxis_title"]
            yaxis_title["text"] = metric
        if height is None:
            height = self.style_dict["height"]
        if width is None:
            width = self.style_dict["width"]
        if hovermode is None:
            hovermode = self.style_dict["hovermode"]

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

        fig.data[0].line.color = self.style_dict["datadrift_historical"]
        fig.data[-1].marker.color = self.style_dict["datadrift_historical"]

        return fig

    def define_style_attributes(self, colors_dict):
        """define_style_attributes allows Eurybia user to change the color of plot

        Parameters
        ----------
        colors_dict: dict
            Dict of the colors used in the different plots

        """
        self.style_dict = define_style(colors_dict)

        if hasattr(self, "pred_colorscale"):
            delattr(self, "pred_colorscale")

    def generate_indicator(
        self,
        fig_value: float,
        min_gauge: float = 0.5,
        max_gauge: float = 1,
        height: float = 300,
        width: float = 500,
        title: str = "Metric",
    ) -> str:
        """Generate a static SVG indicator gauge.

        Args:
            fig_value (float): Value to display on the gauge.
            min_gauge (float, optional): Minimum value of the gauge range. Defaults to 0.5.
            max_gauge (float, optional): Maximum value of the gauge range. Defaults to 1.
            height (float, optional): Height of the plot. Defaults to 300.
            width (float, optional): Width of the plot. Defaults to 500.
            title (str, optional): Title of the plot. Defaults to "Metric".
        Returns:
            str: The generated SVG image.
        """
        if max_gauge <= min_gauge:
            raise ValueError("max_gauge must be greater than min_gauge")

        width = float(width)
        height = float(height)
        center_x = width / 2
        radius = min(width * 0.38, height * 0.48)
        center_y = height * 0.76
        stroke_width = max(20, min(52, height * 0.18))
        ratio = (fig_value - min_gauge) / (max_gauge - min_gauge)
        fill_ratio = min(1, max(0, ratio))
        fill_angle = pi * (1 - fill_ratio)
        fill_width = max(8, stroke_width * 0.55)
        fill_x = center_x + radius * cos(fill_angle)
        fill_y = center_y - radius * sin(fill_angle)
        ticks = []
        tick_labels = []

        for index in range(11):
            angle = pi * (1 - index / 10)
            outer_radius = radius + stroke_width / 2 + 2
            tick_length = 10 if index in (0, 5, 10) else 6
            inner_radius = outer_radius - tick_length
            x_start = center_x + inner_radius * cos(angle)
            y_start = center_y - inner_radius * sin(angle)
            x_end = center_x + outer_radius * cos(angle)
            y_end = center_y - outer_radius * sin(angle)
            ticks.append(
                f'<line class="gauge-tick" x1="{x_start:.2f}" y1="{y_start:.2f}" '
                f'x2="{x_end:.2f}" y2="{y_end:.2f}" stroke="#555" stroke-width="1.5"/>'
            )
            tick_value = min_gauge + (max_gauge - min_gauge) * index / 10
            label_radius = outer_radius + 14
            label_x = center_x + label_radius * cos(angle)
            label_y = center_y - label_radius * sin(angle) + 4
            tick_labels.append(
                f'<text class="gauge-tick-label" x="{label_x:.2f}" y="{label_y:.2f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="15" fill="#555">{tick_value:g}</text>'
            )

        fill_arc = (
            f'<path class="gauge-fill" d="M {center_x - radius:.2f} {center_y:.2f} '
            f'A {radius:.2f} {radius:.2f} 0 0 1 {fill_x:.2f} {fill_y:.2f}" '
            f'fill="none" stroke="#111" stroke-width="{fill_width:.2f}" stroke-linecap="butt"/>'
            if fill_ratio > 0
            else ""
        )
        value_y = center_y - radius * 0.3 + 16
        safe_title = escape(title, quote=True)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" '
            f'viewBox="0 0 {width:g} {height:g}" role="img" aria-label="{safe_title}">'
            f"<title>{safe_title}</title>"
            f'<defs><linearGradient id="gauge-colors" gradientUnits="userSpaceOnUse" '
            f'x1="{center_x - radius:g}" y1="{center_y:g}" x2="{center_x + radius:g}" y2="{center_y:g}">'
            f'<stop offset="0%" stop-color="#008000"/><stop offset="25%" stop-color="#ffff00"/>'
            f'<stop offset="75%" stop-color="#ffa500"/><stop offset="100%" stop-color="#ff0000"/>'
            "</linearGradient></defs>"
            f'<text x="{center_x:g}" y="20" text-anchor="middle" font-family="sans-serif" '
            f'font-size="20" fill="#222">{safe_title}</text>'
            f'<path class="gauge-track" d="M {center_x - radius:.2f} {center_y:.2f} '
            f'A {radius:.2f} {radius:.2f} 0 0 1 {center_x + radius:.2f} {center_y:.2f}" '
            f'fill="none" stroke="url(#gauge-colors)" stroke-width="{stroke_width:.2f}"/>'
            f"{fill_arc}"
            f"{''.join(ticks)}"
            f"{''.join(tick_labels)}"
            f'<text class="gauge-value" x="{center_x:g}" y="{value_y:.2f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="56" font-weight="600" fill="#222">'
            f"{round(fig_value, 2):g}</text>"
            f'<text x="{center_x - radius:g}" y="{center_y + 23:g}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#555">No Drift</text>'
            f'<text x="{center_x + radius:g}" y="{center_y + 23:g}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12" fill="#555">High Drift</text>'
            "</svg>"
        )
