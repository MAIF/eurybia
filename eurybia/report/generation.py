"""
Report generation helper module.
"""
from base64 import b64encode
from datetime import datetime
from typing import Optional

import datapane as dp
import importlib_resources as ir
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from shapash.explainer.smart_explainer import SmartExplainer

from eurybia import SmartDrift
from eurybia.report.project_report import DriftReport


def _load_custom_template(report: dp.Report) -> dp.Report:
    """
    This function feeds a customised html template to Datapane

    Parameters
    ----------
    report : datapane.Report
        Report object
    Returns
    ----------
    datapane.Report
    """
    report._local_writer.assets = ir.files("eurybia.assets")
    logo_img = (report._local_writer.assets / "logo_eurybia_dp.png").read_bytes()
    report._local_writer.logo = f"data:image/png;base64,{b64encode(logo_img).decode('ascii')}"
    template_loader = FileSystemLoader(report._local_writer.assets)
    template_env = Environment(loader=template_loader)
    template_env.globals["include_raw"] = dp.client.api.report.core.include_raw
    report._local_writer.template = template_env.get_template("report_template.html")
    return report


def _get_index(dr: DriftReport, project_info_file: str, config: Optional[dict]) -> dp.Page:
    """
    This function generates and returns a Datapane page containing the Eurybia report index

    Parameters
    ----------
    dr : DriftReport
        DriftReport object
    project_info_file : str
        Path to the file used to display some information about the project in the report.
    config : dict, optional
        Report configuration options.
    Returns
    ----------
    datapane.Page
    """

    eurybia_logo = """
    <html>
        <img style="max-width: 150px; height: auto;" src="https://eurybia.readthedocs.io/en/latest/_images/eurybia-fond-clair.png?raw=true"/>
    </html>
    """

    # main block
    index_block = []

    # Title and logo
    index_block += [dp.Group(dp.HTML(eurybia_logo), dp.Text(f"# {dr.title_story}"), columns=2)]

    if config is not None and "title_description" in config.keys() and config["title_description"] != "":
        raw_title = config["title_description"]
        index_block += [dp.Text(f"## {raw_title}")]
    index_str = "## Eurybia Report contents  \n"

    # Tabs index
    if project_info_file is not None:
        index_str += "- Project information: report context and information  \n"
    index_str += "- Consistency Analysis: highlighting differences between the two datasets  \n"
    index_str += "- Data drift: In-depth data drift analysis \n"

    if dr.smartdrift.data_modeldrift is not None:
        index_str += "- Model drift: In-depth model drift analysis"

    index_block += [dp.Text(index_str)]

    # AUC
    auc_block = dr.smartdrift.plot.generate_indicator(
        fig_value=dr.smartdrift.auc, height=280, width=500, title="Datadrift classifier AUC"
    )

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
        index_block += [dp.Group(auc_block, JS_block, columns=3)]
    else:
        index_block += [dp.Group(auc_block, columns=2)]

    page_index = dp.Page(title="Index", blocks=index_block)
    return page_index


def _dict_to_text_blocks(text_dict, level=1):
    """
    This function recursively explores the dict and returns a Datapane Group containing other groups and text blocks fed with the dict
    Parameters
    ----------
    text_dict: dict
        This dict must contain string as keys, and dicts or strings as values
    level: int = 1
        Recursion level, starting at 1 to allow for direct string manipulation
    Returns
    ----------
    datapane.Group
        Group of blocks
    """
    blocks = []
    text = ""
    for k, v in text_dict.items():
        if isinstance(v, (str, int, float)) or v is None:
            if k.lower() == "date" and v.lower() == "auto":
                v = str(datetime.now())[:-7]
            text += f"**{k}** : {v}  \n"
        elif isinstance(v, dict):
            if text != "":
                blocks.append(dp.Text(text))
                text = ""
            blocks.append(
                dp.Group(dp.Text("#" * min(level, 6) + " " + str(k)), _dict_to_text_blocks(v, level + 1), columns=1)
            )
    if text != "":
        blocks.append(dp.Text(text))
    return dp.Group(blocks=blocks, columns=1)


def _get_project_info(dr: DriftReport) -> dp.Page:
    """
    This function generates and returns a Datapane page from a dict containing dicts and strings

    Parameters
    ----------
    dr : DriftReport
        DriftReport object

    Returns
    ----------
    datapane.Page
    """
    if dr.metadata is None:
        return None
    page_info = dp.Page(
        title="Project information",
        blocks=[_dict_to_text_blocks(dr.metadata)],
    )
    return page_info


def _get_consistency_analysis(dr: DriftReport) -> dp.Page:
    """
    This function generates and returns a Datapane page containing the Eurybia consistency analysis

    Parameters
    ----------
    dr : DriftReport
        DriftReport object

    Returns
    ----------
    datapane.Page
    """
    # Title
    blocks = [dp.Text("# Consistency Analysis")]

    # Manually ignored coluumns
    ignore_cols = pd.DataFrame({"ignore_cols": dr.smartdrift.ignore_cols}).rename(
        columns={"ignore_cols": "Ignored columns"}
    )
    blocks += [
        dp.Text("## Ignored columns in the report (manually excluded)"),
    ]
    if len(ignore_cols) > 0:
        blocks += [dp.Table(data=ignore_cols)]
    else:
        blocks += [dp.Text("- Ignored columns : None.")]

    # Column mismatches
    blocks += [
        dp.Text("## Consistency checks: column match between the 2 datasets."),
        dp.Text(
            """
            The columns identified in this section have been automatically removed from this analysis.
            Their presence would always be sufficient for the datadrift classifier to perfectly discriminate the two datasets (maximal data drift, AUC=1).
            """
        ),
    ]
    for k, v in dr.smartdrift.pb_cols.items():
        if len(v) > 0:
            blocks += [dp.Table(data=pd.DataFrame(v).transpose())]
        else:
            blocks += [dp.Text(f"- No {k.lower()} have been detected.")]

    blocks += [
        dp.Text("### Unique values identified:"),
        dp.Text(
            """
            This section displays categorical features in which unique values differ.
            This analysis has been performed on unstratified samples of the baseline and current datasets.
            Missing or added unique values can be caused by this sampling.
            Columns identified in this section have been kept for the analysis.
            """
        ),
    ]
    if len(dr.smartdrift.err_mods) > 0:
        blocks += [
            dp.Table(
                data=pd.DataFrame(dr.smartdrift.err_mods)
                .rename(columns={"err_mods": "Modalities present in one dataset and absent in the other :"})
                .transpose()
            )
        ]
    else:
        blocks += [dp.Text("- No modalities have been detected as present in one dataset and absent in the other.")]

    page_consistency = dp.Page(title="Consistency Analysis", blocks=blocks)
    return page_consistency


def _get_datadrift(dr: DriftReport) -> dp.Page:
    """
    This function generates and returns a Datapane page containing the Eurybia data drift analysis

    Parameters
    ----------
    dr : DriftReport
        DriftReport object

    Returns
    ----------
    datapane.Page
    """
    # Loop for save in list plots of display analysis
    plot_dataset_analysis = []
    table_dataset_analysis = []
    fig_list, labels, table_list = dr.display_dataset_analysis(global_analysis=False)["univariate"]
    for i in range(len(labels)):
        plot_dataset_analysis.append(dp.Plot(fig_list[i], label=labels[i]))
        table_dataset_analysis.append(dp.Table(table_list[i], label=labels[i]))

    # Loop for save in list plots of display analysis
    plot_datadrift_contribution = []
    fig_list, labels = dr.display_model_contribution()
    for i in range(len(labels)):
        plot_datadrift_contribution.append(dp.Plot(fig_list[i], label=labels[i]))
    blocks = [
        dp.Text("# Data drift"),
        dp.Text(
            """The data drift detection methodology is based on the ability of a model classifier to identify whether
        a sample belongs to one or another dataset.
        For this purpose a target (0) is assigned to the baseline dataset and a second target (1) to the current dataset.
        A classification model (catboost) is trained to predict this target.
        As such, the data drift classifier performance is directly related to the difference between two datasets.
        A marked difference will lead to an easy classification (final AUC close to 1).
        Oppositely, highly similars datasets will lead to poor data drift classifier performance (final AUC close to 0.5)."""
        ),
        dp.Text("## Detecting data drift"),
        dp.Text("### Datadrift classifier model perfomances"),
        dp.Text(
            """The closer your AUC is from 0.5 the less your data drifted.
             The closer your AUC is from 1 the more your data drifted"""
        ),
        dp.Plot(
            dr.smartdrift.plot.generate_indicator(
                fig_value=dr.smartdrift.auc, height=300, width=500, title="Datadrift classifier AUC"
            )
        ),
        dp.Text("## Importance of features in data drift"),
        dp.Text("### Global feature importance plot"),
        dp.Text(
            """Bar chart representing the feature importance of each feature for the datadrift classifier.
             This parameter is a direct measure of the importance of a feature to perform the classification."""
        ),
        dp.Plot(dr.explainer.plot.features_importance()),
    ]
    if dr.smartdrift.deployed_model is not None:
        blocks += [
            dp.Text("### Feature importance overview"),
            dp.Text(
                """Scatter plot depicting, for each feature, the feature importance of the deployed model as a function of the datadrift classifier
                 feature importance. This graph thus highlight the real importance of a data drift for the deployed model classification.
                Interpretation based on graphical feature location:
            - Top left : Feature highly important for the deployed model and with low data drift
            - Bottom left : Feature with moderated importance for the deployed model and with low data drift
            - Bottom right : Feature with moderated importance for the deployed model but with high data drift.
            This feature might require your attention.
            - Top right : Feature highly important for the deployed model and high drift. This feature requires your attention.
            """
            ),
            dp.Plot(dr.smartdrift.plot.scatter_feature_importance()),
        ]
    blocks += [
        dp.Text("## Dataset analysis"),
        dp.Text(
            """This section provides numerical and graphical analysis of the 2 datasets distributions,
            making easier the study of the most important variable for drift detection."""
        ),
        dp.Text("### Global analysis"),
        dp.Table(dr._display_dataset_analysis_global()),
        dp.Text("### Univariate analysis"),
        dp.Text(
            """Bar chart showing the unique values distribution of a feature.
        Using the drop-down menu, it is possible to select the feature of interest.
        Features are sorted according to their respective importance in the datadrift classifier.
        For categorical features, the possible values are sorted by descending difference between the two datasets."""
        ),
        dp.Select(blocks=plot_dataset_analysis),
        dp.Select(blocks=table_dataset_analysis),
    ]
    if dr.smartdrift.deployed_model is not None:
        blocks += [
            dp.Text("### Distribution of predicted values"),
            dp.Text(
                "Histogram density showing the distributions of the production model outputs on both baseline and current datasets."
            ),
            dp.Plot(
                dr.smartdrift.plot.generate_fig_univariate(df_all=dr.smartdrift.df_predict, col="Score", hue="dataset")
            ),
            dp.Text(
                """Jensen Shannon Divergence (JSD). The JSD measures the effect of a data drift on the deployed model performance.
            A value close to 0 indicates similar data distributions, while a value close to 1 tend to indicate distinct data distributions
            with a negative effect on the deployed model performance."""
            ),
            dr.smartdrift.plot.generate_indicator(
                fig_value=dr.smartdrift.js_divergence,
                height=280,
                width=500,
                title="Jensen Shannon Datadrift",
                min_gauge=0,
                max_gauge=0.2,
            ),
        ]
    blocks += [
        dp.Text("## Feature contribution on data drift's detection"),
        dp.Text(
            """This graph represents the contribution of a variable to the data drift detection.
        This representation constitutes a support to understand the drift when the analysis of the dataset is unclear.
        In the drop-down menu, features are sorted by importance in the data drift detection."""
        ),
        dp.Select(blocks=plot_datadrift_contribution),
    ]
    if dr.smartdrift.historical_auc is not None:
        blocks += [
            dp.Text("## Historical Data drift"),
            dp.Text(
                "Line chart showing the metrics evolution of the datadrift classifier over the given period of time."
            ),
            dp.Plot(dr.smartdrift.plot.generate_historical_datadrift_metric()),
        ]
    page_datadrift = dp.Page(title="Data drift", blocks=blocks)
    return page_datadrift


def _get_modeldrift(dr: DriftReport) -> dp.Page:
    """
    This function generates and returns a Datapane page containing the Eurybia model drift analysis

    Parameters
    ----------
    dr : DriftReport
        DriftReport object

    Returns
    ----------
    datapane.Page
    """
    # Loop for save in list plots of display model drift
    if dr.smartdrift.data_modeldrift is not None:
        plot_modeldrift = []
        fig_list, labels = dr.display_data_modeldrift()
        if labels == []:
            plot_modeldrift = dp.Plot(fig_list[0])
            modeldrift_plot = plot_modeldrift
        else:
            for i in range(len(labels)):
                plot_modeldrift.append(dp.Plot(fig_list[i], label=labels[i]))
            modeldrift_plot = dp.Select(blocks=plot_modeldrift, label="reference_columns")
    else:
        modeldrift_plot = dp.Text("## Smartdrift.data_modeldrift is None")
    blocks = [
        dp.Text("# Model drift"),
        dp.Text(
            """This section provides support to monitor the production model's performance over time.
    This requires the performance history as input."""
        ),
        dp.Text("## Performance evolution of the deployed model"),
        dp.Text("Line chart of deployed model performances as a function of time"),
        modeldrift_plot,
    ]
    page_modeldrift = dp.Page(title="Model drift", blocks=blocks)
    return page_modeldrift


def execute_report(
    smartdrift: SmartDrift,
    explainer: SmartExplainer,
    project_info_file: str,
    output_file: str,
    config: Optional[dict] = None,
):
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
    config : dict, optional
        Report configuration options.
    output_file : str
            Path to the HTML file to write
    """

    if config is None:
        config = {}

    dr = DriftReport(
        smartdrift=smartdrift,
        explainer=explainer,  # rename to match kwarg
        project_info_file=project_info_file,
        config=config,
    )

    pages = []
    pages.append(_get_index(dr, project_info_file, config))
    if project_info_file is not None:
        pages.append(_get_project_info(dr))
    pages.append(_get_consistency_analysis(dr))
    pages.append(_get_datadrift(dr))
    if dr.smartdrift.data_modeldrift is not None:
        pages.append(_get_modeldrift(dr))

    report = dp.Report(blocks=pages)
    report = _load_custom_template(report)
    report._save(
        path=output_file, open=False, formatting=dp.ReportFormatting(light_prose=False, width=dp.ReportWidth.MEDIUM)
    )
