"""
Report generation helper module.
"""

from datetime import datetime
from typing import Any, Optional

import panel as pn
import pandas as pd
from shapash.explainer.smart_explainer import SmartExplainer

from eurybia import SmartDrift
from eurybia.report.project_report import DriftReport
from eurybia.report.properties import report_css, report_jscallback, report_text


pn.extension("plotly")


def get_index_panel(dr: DriftReport, project_info_file: str, config_report: Optional[dict]) -> pn.Column:
    parts = []
    header_logo = pn.pane.PNG(
        "https://eurybia.readthedocs.io/en/latest/_images/eurybia-fond-clair.png?raw=true",
        styles={"max-width": "150px", "height": "auto"},
    )
    header_title = pn.pane.Markdown(f"# {dr.title_story}")
    header = pn.Row(header_logo, header_title)
    parts.append(header)

    if (
        config_report is not None
        and "title_description" in config_report.keys()
        and config_report["title_description"] != ""
    ):
        raw_title = config_report["title_description"]
        parts.append(pn.pane.Markdown(f"## {raw_title}"))

    content_parts = ["## Eurybia Report contents"]
    if project_info_file is not None:
        content_parts.append(report_text["Index"]["01"])
    content_parts.append(report_text["Index"]["02"])
    content_parts.append(report_text["Index"]["03"])
    if dr.smartdrift.data_modeldrift is not None:
        content_parts.append(report_text["Index"]["04"])
    content = pn.pane.Markdown("\n".join(content_parts))
    parts.append(content)

    # AUC
    auc_block = dr.smartdrift.plot.generate_indicator(
        fig_value=dr.smartdrift.auc, height=280, width=500, title="Datadrift classifier AUC"
    )
    auc_indicator = pn.pane.Plotly(auc_block)

    # Jensen-Shannon
    if dr.smartdrift.deployed_model is not None:
        JS_block = dr.smartdrift.plot.generate_indicator(
            fig_value=dr.smartdrift.js_divergence,
            height=280,
            width=500,
            title="Jensen Shannon Datadrift",
            min_gauge=0,
            max_gauge=0.2,
        )
        js_indicator = pn.pane.Plotly(JS_block)
        indicators = pn.Row(auc_indicator, js_indicator)

    else:
        indicators = pn.Row(auc_indicator)
    parts.append(indicators)

    return pn.Column(*parts, name="Index")


def dict_to_text_blocks(text_dict: dict, level: int = 1) -> pn.Column:
    """
    This function recursively explores the dict and returns a Panel Column containing
    other groups and text blocks fed with the dict
    Parameters
    ----------
    text_dict: dict
        This dict must contain string as keys, and dicts or strings as values
    level: int = 1
        Recursion level, starting at 1 to allow for direct string manipulation
    Returns
    ----------
    pn.Column
        Column of blocks
    """
    blocks = []
    text = ""
    for k, v in text_dict.items():
        if isinstance(v, (str, int, float)) or v is None:
            if k.lower() == "date" and isinstance(v, str) and v.lower() == "auto":
                v = str(datetime.now())[:-7]
            text += f"**{k}** : {v}  \n"
        elif isinstance(v, dict):
            if text != "":
                blocks.append(pn.pane.Markdown(text))
                text = ""
            blocks.append(
                pn.Column(pn.pane.Markdown("#" * min(level, 6) + " " + str(k)), dict_to_text_blocks(v, level + 1))
            )
    if text != "":
        blocks.append(pn.pane.Markdown(text))
    return pn.Column(*blocks)


def get_project_information_panel(dr: DriftReport) -> Optional[pn.Column]:
    if dr.metadata is None:
        return None
    blocks = dict_to_text_blocks(dr.metadata)
    return pn.Column(*blocks, name="Project information", styles=dict(display="none"))


def get_consistency_analysis_panel(dr: DriftReport) -> pn.Column:

    # Title
    blocks = [pn.pane.Markdown("# Consistency Analysis")]

    # Manually ignored coluumns
    ignore_cols = pd.DataFrame({"ignore_cols": dr.smartdrift.ignore_cols}).rename(
        columns={"ignore_cols": "Ignored columns"}
    )
    blocks += [
        pn.pane.Markdown("## Ignored columns in the report (manually excluded)"),
    ]
    if len(ignore_cols) > 0:
        blocks += [pn.pane.DataFrame(ignore_cols)]
    else:
        blocks += [pn.pane.Markdown("- Ignored columns : None.")]

    # Column mismatches
    blocks += [
        pn.pane.Markdown("## Consistency checks: column match between the 2 datasets."),
        pn.pane.Markdown(report_text["Consistency analysis"]["01"]),
    ]
    for k, v in dr.smartdrift.pb_cols.items():
        if len(v) > 0:
            blocks += [pn.pane.DataFrame(pd.DataFrame(v).transpose())]
        else:
            blocks += [pn.pane.Markdown(f"- No {k.lower()} have been detected.")]

    blocks += [
        pn.pane.Markdown("###  Unique values identified"),
        pn.pane.Markdown(report_text["Consistency analysis"]["02"]),
    ]
    if len(dr.smartdrift.err_mods) > 0:
        blocks += [
            pn.pane.DataFrame(
                pd.DataFrame(dr.smartdrift.err_mods)
                .rename(columns={"err_mods": "Modalities present in one dataset and absent in the other :"})
                .transpose(),
            )
        ]
    else:
        blocks += [
            pn.pane.Markdown("- No modalities have been detected as present in one dataset and absent in the other.")
        ]

    return pn.Column(*blocks, name="Consistency Analysis", styles=dict(display="none"))


def get_data_drift_detecting(dr: DriftReport) -> list:
    blocks = [pn.pane.Markdown("## Detecting data drift")]
    blocks.append(pn.pane.Markdown("### Datadrift classifier model perfomances"))
    blocks.append(pn.pane.Markdown(report_text["Data drift"]["02"]))
    auc = dr.smartdrift.plot.generate_indicator(
        fig_value=dr.smartdrift.auc, height=300, width=500, title="Datadrift classifier AUC"
    )
    blocks.append(pn.pane.Plotly(auc))
    return blocks


def get_data_drift_features_importance(dr: DriftReport) -> list:
    blocks = [pn.pane.Markdown("## Importance of features in data drift")]
    blocks.append(pn.pane.Markdown("### Global feature importance plot"))
    blocks.append(pn.pane.Markdown(report_text["Data drift"]["03"]))
    fig_features_importance = dr.explainer.plot.features_importance()
    fig_features_importance.update_layout(width=1240)
    blocks.append(pn.pane.Plotly(fig_features_importance))
    # blocks.append(pn.pane.Plotly(dr.explainer.plot.features_importance(width=1240)))
    if dr.smartdrift.deployed_model is not None:
        fig_scatter_feature_importance = dr.smartdrift.plot.scatter_feature_importance()
        fig_scatter_feature_importance.update_layout(width=1240)
        blocks += [
            pn.pane.Markdown("### Feature importance overview"),
            pn.pane.Markdown(report_text["Data drift"]["04"]),
            pn.pane.Plotly(fig_scatter_feature_importance)
            # pn.pane.Plotly(dr.smartdrift.plot.scatter_feature_importance()),
        ]
    return blocks


def get_data_drift_dataset_analysis(dr: DriftReport) -> list:
    blocks = [pn.pane.Markdown("## Dataset analysis")]
    blocks += [
        pn.pane.Markdown(report_text["Data drift"]["05"]),
        pn.pane.Markdown("### Global analysis"),
        pn.pane.DataFrame(dr._display_dataset_analysis_global()),
    ]

    if dr.smartdrift.deployed_model is not None:
        fig_01 = dr.smartdrift.plot.generate_fig_univariate(df_all=dr.smartdrift.df_predict, col="Score", hue="dataset")
        fig_01.update_layout(width=1240)
        blocks += [
            pn.pane.Markdown("### Distribution of predicted values"),
            pn.pane.Markdown(report_text["Data drift"]["06"]),
            pn.pane.Plotly(fig_01),
            pn.pane.Markdown(report_text["Data drift"]["07"]),
        ]
        js_fig = dr.smartdrift.plot.generate_indicator(
            fig_value=dr.smartdrift.js_divergence,
            height=280,
            width=500,
            title="Jensen Shannon Datadrift",
            min_gauge=0,
            max_gauge=0.2,
        )
        blocks += [pn.pane.Plotly(js_fig)]

    blocks += [
        pn.pane.Markdown("### Univariate analysis"),
        pn.pane.Markdown(report_text["Data drift"]["08"])
    ]
    plot_datadrift_contribution = {}
    frame_datadrift_contribution = {}
    plot_feature_contribution = {}
    fig_contribution_list, labels = dr.display_model_contribution()
    fig_list, labels, table_list = dr.display_dataset_analysis(global_analysis=False)["univariate"]
    for i in range(len(labels)):
        fig_list[i].update_layout(width=1240)
        fig_contribution_list[i].update_layout(width=1240)
        plot_datadrift_contribution[labels[i]] = pn.pane.Plotly(fig_list[i])
        frame_datadrift_contribution[labels[i]] = pn.pane.DataFrame(table_list[i])
        plot_feature_contribution[labels[i]] = pn.pane.Plotly(fig_contribution_list[i])
    plot_dataset_panel = pn.Column(plot_datadrift_contribution[labels[0]])
    frame_dataset_panel = pn.Column(plot_datadrift_contribution[labels[0]])
    feature_contribution_panel = pn.Column(plot_feature_contribution[labels[0]])
    feature_select = pn.widgets.Select(value=labels[0], options=list(plot_datadrift_contribution.keys()))

    def update_feature(event: Any) -> None:
        plot_dataset_panel[0] = plot_datadrift_contribution[feature_select.value]
        frame_dataset_panel[0] = frame_datadrift_contribution[feature_select.value]
        feature_contribution_panel[0] = plot_feature_contribution[feature_select.value]

    feature_select.param.watch(update_feature, "value")
    blocks += [
        feature_select,
        pn.pane.Markdown("#### Distribution of feature"),
        plot_dataset_panel,
        frame_dataset_panel,
        pn.pane.Markdown("#### Contribution of feature on data drift's detection"),
        pn.pane.Markdown(report_text["Data drift"]["09"]),
        feature_contribution_panel,
    ]

    return blocks


def get_data_drift_panel(dr: DriftReport) -> pn.Column:
    blocks = [
        pn.pane.Markdown("# Data drift"),
        pn.pane.Markdown(report_text["Data drift"]["01"]),
    ]
    blocks += get_data_drift_detecting(dr)
    blocks += get_data_drift_features_importance(dr)
    blocks += get_data_drift_dataset_analysis(dr)

    return pn.Column(*blocks, name="Data drift", styles=dict(display="none"))


def get_model_drift_panel(dr: DriftReport) -> pn.Column:
    """
    This function generates and returns a Panel Column page containing the Eurybia model drift analysis

    Parameters
    ----------
    dr : DriftReport
        DriftReport object

    Returns
    ----------
    pn.Column
    """

    # Loop for save in list plots of display model drift
    modeldrift_plot = None
    if dr.smartdrift.data_modeldrift is not None:
        fig_list, labels = dr.display_data_modeldrift()
        if labels == []:
            fig_list[0].update_layout(width=1240)
            modeldrift_plot = pn.pane.Plotly(fig_list[0])
        else:
            elements = []
            for i in range(len(labels)):
                fig_list[i].update_layout(width=1100)
                elements.append(pn.Column(pn.pane.Plotly(fig_list[i]), name=labels[i], styles={"text-align": "left"}))
            plot_modeldrift_panel = pn.Tabs(*elements, tabs_location="left")
    else:
        modeldrift_plot = pn.pane.Markdown("## Smartdrift.data_modeldrift is None")
    blocks = [
        pn.pane.Markdown("# Model drift"),
        pn.pane.Markdown(report_text["Model drift"]["01"]),
        pn.pane.Markdown("## Performance evolution of the deployed model"),
        pn.pane.Markdown(report_text["Model drift"]["02"]),
    ]
    if modeldrift_plot is not None:
        blocks += [modeldrift_plot]
    else:
        blocks += [plot_modeldrift_panel]
    return pn.Column(*blocks, name="Model drift", styles=dict(display="none"), css_classes=['modeldrift-panel'])


def execute_report(
    smartdrift: SmartDrift,
    explainer: SmartExplainer,
    project_info_file: str,
    output_file: str,
    config_report: Optional[dict] = {},
) -> None:
    """
    Creates the report

    Parameters
    ----------
    smartdrift : eurybia.core.smartdrift.SmartDrift object
        Compiled SmartDrift class
    explainer : shapash.explainer.smart_explainer.SmartExplainer object
        Compiled shapash explainer.
    project_info_file : str
        Path to the file used to display some information about the project in the report.
    config_report : dict, optional
        Report configuration options.
    output_file : str
            Path to the HTML file to write
    """
    dr = DriftReport(
        smartdrift=smartdrift,
        explainer=explainer,
        project_info_file=project_info_file,
        config_report=config_report,
    )

    tab_list = []
    tab_list.append(get_index_panel(dr, project_info_file, config_report))
    if project_info_file is not None:
        tab_list.append(get_project_information_panel(dr))
    tab_list.append(get_consistency_analysis_panel(dr))
    tab_list.append(get_data_drift_panel(dr))
    if dr.smartdrift.data_modeldrift is not None:
        tab_list.append(get_model_drift_panel(dr))

    pn.config.raw_css.append(report_css)
    report = pn.Tabs(*tab_list, css_classes=['main-report'])
    report.jscallback(args={"active": report}, active=report_jscallback)
    report.save(output_file, embed=True)
