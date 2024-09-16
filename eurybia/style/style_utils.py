"""
functions for loading and manipulating colors
"""

import json
import os


def colors_loading():
    """
    colors_loading allows Eurybia to load a json file which contains different
    palettes of colors that can be used in the plot
    Returns
    -------
    dict:
        contains all available pallets
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    jsonfile = os.path.join(current_path, "colors.json")
    with open(jsonfile) as openfile:
        colors_dic = json.load(openfile)
    return colors_dic


def select_palette(colors_dic, palette_name):
    """
    colors_loading allows Eurybia to load a json file which contains different
    palettes of colors that can be used in the plot
    Parameters
    ----------
    colors_dic : dict
        dictionnary with every palettes
    palette_name : String
        name of the palette
    Returns
    -------
    dict:
        contains colors of one palette
    """
    if palette_name not in colors_dic.keys():
        raise ValueError(f"Palette {palette_name} not found.")
    return colors_dic[palette_name]


def define_style(palette):
    """
    the define_style function is a function that uses a palette
    to define the different styles used in the different outputs
    of Eurybia
    Parameters
    ----------
    palette : dict
        contains colors of one palette
    Returns
    -------
    dict :
        contains different style elements
    """
    style_dict = dict()
    style_dict["dict_title"] = {
        "xanchor": "center",
        "yanchor": "middle",
        "x": 0.5,
        "y": 0.9,
        "font": {"size": 24, "family": "Arial", "color": palette["title_color"]},
    }

    featureimp_bar = palette["featureimp_bar"]
    style_dict["dict_featimp_colors"] = {
        "color": featureimp_bar,
        "line": {"color": palette["featureimp_bar"], "width": 0.5},
    }
    style_dict["featureimp_groups"] = list(palette["featureimp_groups"].values())

    style_dict["dict_xaxis_title"] = {"font": {"size": 16, "family": "Arial Black", "color": palette["axis_color"]}}
    style_dict["dict_yaxis_title"] = {"font": {"size": 16, "family": "Arial Black", "color": palette["axis_color"]}}
    style_dict["dict_xaxis"] = dict(
        linecolor="#BCCCDC",
        showspikes=True,  # Show spike line for X-axis
        # Format spike
        spikethickness=2,
        spikedash="dot",
        spikecolor="#999999",
        spikemode="across",
    )
    style_dict["dict_xaxis_continuous"] = {
        "font": {"size": 16, "family": "Arial Black", "color": palette["axis_color"]}
    }
    style_dict["dict_yaxis_continuous"] = {
        "font": {"size": 16, "family": "Arial Black", "color": palette["title_color"]},
        "text": "Density",
    }
    style_dict["dict_legend"] = {"title": ""}
    style_dict["template"] = "simple_white"
    style_dict["height"] = 600
    style_dict["width"] = 900
    style_dict["univariate_cat_bar"] = palette["univariate_cat_bar"]
    style_dict["univariate_cont_bar"] = palette["univariate_cont_bar"]
    style_dict["datadrift_historical"] = palette["datadrift_historical"]
    style_dict["scatter_plot"] = palette["scatter_plot"]
    style_dict["scatter_line"] = palette["scatter_line"]
    style_dict["featimportance_colorscale"] = palette["featimportance_colorscale"]
    style_dict["contrib_colorscale"] = palette["contrib_colorscale"]
    style_dict["hovermode"] = "closest"
    return style_dict
