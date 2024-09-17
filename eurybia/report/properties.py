from typing import Any

report_text: dict[str, Any] = {
    "Index": {
        "01": "- Project information: report context and information",
        "02": "- Consistency Analysis: highlighting differences between the two datasets",
        "03": "- Data drift: In-depth data drift analysis",
        "04": "- Model drift: In-depth model drift analysis",
    },
    "Consistency analysis": {
        "01": (
            "The columns identified in this section have been automatically removed from this analysis. "
            "Their presence would always be sufficient for the datadrift classifier to perfectly discriminate "
            "the two datasets (maximal data drift, AUC=1)."
        ),
        "02": (
            "This section displays categorical features in which unique values differ. "
            "This analysis has been performed on unstratified samples of the baseline and current datasets. "
            "Missing or added unique values can be caused by this sampling. "
            "Columns identified in this section have been kept for the analysis."
        ),
    },
    "Data drift": {
        "01": (
            "The data drift detection methodology is based on the ability of a model classifier to identify "
            "whether a sample belongs to one or another dataset. For this purpose a target (0) is assigned "
            "to the baseline dataset and a second target (1) to the current dataset. "
            "A classification model (catboost) is trained to predict this target. As such, the data drift "
            "classifier performance is directly related to the difference between two datasets. A marked difference "
            " will lead to an easy classification (final AUC close to 1). Oppositely, highly similars datasets "
            " will lead to poor data drift classifier performance (final AUC close to 0.5)."
        ),
        "02": (
            "The closer your AUC is from 0.5 the less your data drifted. "
            "The closer your AUC is from 1 the more your data drifted."
        ),
        "03": (
            "Bar chart representing the feature importance of each feature for the datadrift classifier. "
            "This parameter is a direct measure of the importance of a feature to perform the classification."
        ),
        "04": (
            "Scatter plot depicting, for each feature, the feature importance of the deployed model "
            "as a function of the datadrift classifier "
            "feature importance. This graph thus highlight the real importance of a data drift "
            "for the deployed model classification. "
            "Interpretation based on graphical feature location:\n"
            "- Top left : Feature highly important for the deployed model and with low data drift\n"
            "- Bottom left : Feature with moderated importance for the deployed model and with low data drift.\n"
            "- Bottom right : Feature with moderated importance for the deployed model but with high data drift. "
            "This feature might require your attention.\n"
            "- Top right : Feature highly important for the deployed model and high drift. "
            "This feature requires your attention."
        ),
        "05": (
            "This section provides numerical and graphical analysis of the 2 datasets distributions, "
            "making easier the study of the most important variable for drift detection."
        ),
        "06": (
            "Histogram density showing the distributions of the production model outputs on "
            "both baseline and current datasets."
        ),
        "08": (
            "Jensen Shannon Divergence (JSD). "
            "The JSD measures the effect of a data drift on the deployed model performance. "
            "A value close to 0 indicates similar data distributions, while a value close to 1 "
            "tend to indicate distinct data distributions with a negative effect on the deployed model performance."
        ),
        "07": (
            "Bar chart showing the unique values distribution of a feature. "
            "Using the drop-down menu, it is possible to select the feature of interest. "
            "Features are sorted according to their respective importance in the datadrift classifier. "
            "For categorical features, the possible values are sorted by descending difference "
            "between the two datasets."
        ),
        "09": (
            "This graph represents the contribution of a variable to the data drift detection. "
            "This representation constitutes a support to understand the drift "
            "when the analysis of the dataset is unclear."
        ),
        "10": (
            "This graph represents the interactions between couple of variable to the data drift detection. "
            "This representation constitutes a support to understand the drift "
            "when the analysis of the dataset is unclear."
        ),
        "11": ("Line chart showing the metrics evolution of the datadrift classifier over the given period of time."),
    },
    "Model drift": {
        "01": (
            "This section provides support to monitor the production model's performance over time. "
            "This requires the performance history as input."
        ),
        "02": ("Line chart of deployed model performances as a function of time."),
    },
}

report_css: str = """
.bk-tab {
    background-color: white !important;
    font-size: 0.875rem;
    padding-top: 1em;
    padding-bottom: 1em;
    border-width: 3px !important;
    border-style: solid !important;
    border-color: white !important;
    color: #6b7280;
    margin-left: 1.5em;
    margin-right: 1.5em;
    text-align: left;
}

.bk-tab.bk-active {
    border-bottom-color: #4e46e5 !important;
    color: #101010;
}

.bk-header {
    position: sticky;
    top: 0;
    z-index: 100;
    width: 100%;
    background-color: white;
}

:host(.bk-above) .bk-tab:first-child {
    margin-left: 5rem !important;
}

.bk-above {
    width: 100%;
}

.bk-Row {
    width: 100%;
    display: grid;
    grid-template-columns: auto auto;
}

.bk-panel-models-layout-Column {
    margin-left: auto;
    margin-right: auto;
    width: 100%;
    max-width: 1280px;
    padding-left: 1rem;
    padding-right: 1rem;
}

.bk-panel-models-markup-HTML {
    width: 100%;
}

.bk-panel-models-plotly-PlotlyPlot {
    width: 100%;
    display: flex;
    justify-content: center;
}

h1 {
    font-size: 2.25rem;
    line-height: 1.33;
    font-weight: 700;
}

h2 {
    font-size: 1.5rem;
    line-height: 1.33;
    font-weight: 700;
}

h3 {
    font-size: 1.25rem;
    line-height: 1.6;
    font-weight: 600;
}

h4 {
    font-size: 1rem;
    line-height: 1.6;
    font-weight: 600;
}

ul {
    padding-left: 1.625em;
}

li {
    font-size: 1rem;
    line-height: 1.75;
    padding-left: 0.375em;
    margin-top: 0.5em;
    margin-bottom: 0.5em;
}

p {
    font-size: 1rem;
}

th {
    vertical-align: bottom;
    text-align: center !important;
    font-weight: 600;
    font-size: 1rem;
}

td {
    vertical-align: top !important;
    text-align: center !important;
    font-size: 1rem;
}

select {
    font-size: 1rem;
    font-weight: 700;
}


.hidden {
    display: none;
}

"""

report_jscallback: str = """
console.log("callback called");
var active_tab = active.active;
var top = document.getElementsByTagName('div')[1];
var elements = top.shadowRoot.children;
var current_tab = 0;
for (var i=0 ; i<elements.length ; i++) {
    var tagName = elements[i].tagName;
    if (tagName === 'DIV') {
        var className = elements[i].className;
        if (className !== "bk-header" && current_tab !== active_tab) {
            elements[i].style.removeProperty('visibility');
            elements[i].style['display'] = 'none';
        } else {
            elements[i].style.removeProperty("display");
        }
        current_tab = current_tab + 1;
    }
}
window.scrollTo(0, 0);
"""

select_callback: str = """
console.log("Tab = " + tab);
console.log("Key = " + key);
var f_class = this.value.replace(" ", "-").toLowerCase();
console.log("Feature = " + f_class);
var elts = document.querySelectorAll(".bk-above");
console.log("Elts : " + elts + " - length: " + elts.length);
for (let i=0; i<elts.length; i++) {
  var columns = elts[i].shadowRoot.querySelectorAll(tab);
  console.log("Columns : " + columns + " - length: " + columns.length);
  for (let j=0; j<columns.length; j++) {
    var nodes = columns[j].shadowRoot.querySelectorAll(key);
    console.log("Nodes : " + nodes + " - length: " + nodes.length);
    for (let k=0; k<nodes.length; k++) {
      if (nodes[k].classList.contains(f_class)) {
        nodes[k].classList.remove("hidden");
      } else {
        nodes[k].classList.add("hidden");
      }
    }
  }
}
"""
