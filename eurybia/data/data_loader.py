"""
Data loader module
"""

import json
import os
from urllib.request import urlretrieve

import pandas as pd


def data_loading(dataset):
    """
    data_loading allows Eurybia user to try the library with small but clear datasets.
    Titanic, house_prices or us_car_accident data.

    Example
    ----------
    >>> from eurybia.data.data_loader import data_loading
    >>> house_df, house_dict = data_loading('house_prices')

    Parameters
    ----------
    dataset : String
        Dataset's name to return.
         - 'titanic'
         - 'house_prices'
         - 'us_car_accident'

    Returns
    -------
    data : pandas.DataFrame
        Dataset required
    dict : (Dictionnary, Optional)
        If exist, columns labels dictionnary associated to the dataset.
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    if dataset == "house_prices":
        if os.path.isfile(current_path + "/house_prices_dataset.csv") is False:
            github_data_url = "https://github.com/MAIF/eurybia/raw/master/eurybia/data/"
            urlretrieve(
                github_data_url + "house_prices_dataset.csv", filename=current_path + "/house_prices_dataset.csv"
            )
            urlretrieve(
                github_data_url + "house_prices_labels.json", filename=current_path + "/house_prices_labels.json"
            )
        data_house_prices_path = os.path.join(current_path, "house_prices_dataset.csv")
        dict_house_prices_path = os.path.join(current_path, "house_prices_labels.json")
        data = pd.read_csv(data_house_prices_path, header=0, index_col=0, engine="python")
        with open(dict_house_prices_path) as openfile2:
            dic = json.load(openfile2)
        return data, dic

    elif dataset == "titanic":
        if os.path.isfile(current_path + "/titanicdata.csv") is False:
            github_data_url = "https://github.com/MAIF/eurybia/raw/master/eurybia/data/"
            urlretrieve(github_data_url + "titanicdata.csv", filename=current_path + "/titanicdata.csv")
        data_titanic_path = os.path.join(current_path, "titanicdata.csv")
        data = pd.read_csv(data_titanic_path, header=0, index_col=0, engine="python")
        return data

    elif dataset == "us_car_accident":
        if os.path.isfile(current_path + "/US_Accidents_extract.csv") is False:
            github_data_url = "https://github.com/MAIF/eurybia/raw/master/eurybia/data/"
            urlretrieve(
                github_data_url + "US_Accidents_extract.csv", filename=current_path + "/US_Accidents_extract.csv"
            )
        data_us_car_path = os.path.join(current_path, "US_Accidents_extract.csv")
        data = pd.read_csv(data_us_car_path, engine="python")
        return data

    else:
        raise ValueError("Dataset not found. Check the docstring for available values")
