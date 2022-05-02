<p align="center">
<img src="https://raw.githubusercontent.com/MAIF/eurybia/master/docs/_static/eurybia-fond-clair.png" width="300" title="eurybia-logo">
</p>

<p align="center">
  <!-- Tests -->
  <a href="https://github.com/MAIF/eurybia/workflows/Build%20%26%20Test/badge.svg">
    <img src="https://github.com/MAIF/eurybia/workflows/Build%20%26%20Test/badge.svg" alt="tests">
  </a>
  <!-- PyPi -->
  <a href="https://img.shields.io/pypi/v/eurybia">
    <img src="https://img.shields.io/pypi/v/eurybia" alt="pypi">
  </a>
  <!-- Python Version -->
  <a href="https://img.shields.io/pypi/pyversions/eurybia">
    <img src="https://img.shields.io/pypi/pyversions/eurybia" alt="pyversion">
  </a>
  <!-- License -->
  <a href="https://img.shields.io/pypi/l/eurybia">
    <img src="https://img.shields.io/pypi/l/eurybia" alt="license">
  </a>
  <!-- Doc -->
  <a href="https://eurybia.readthedocs.io/en/latest/">
    <img src="https://readthedocs.org/projects/eurybia/badge/?version=latest" alt="doc">
  </a>
</p>


## 🔍 Overview


**Eurybia** is a Python library which aims to help in detecting drift and validate data before putting a model in production. Eurybia secures deployment of a model in production and ensures that the model does not drift over time. Thus, it contributes for a better model monitoring, model auditing and more generally AI governance.

<p align="center">
  <img src="https://github.com/MAIF/eurybia/blob/master/docs/_static/lifecycle_ml_models.png?raw=true" width="90%" />
</p>

Let's respectively name features, target and prediction of a model X, Y and P(X, Y). P(X, Y) can be decompose as : P(X, Y) = P(Y|X)P(X), with P(Y|X), the conditional probability of ouput given the model features, and P(X) the probability density of the model features.

Data Validation : Validate that data used for production prediction are similar to training data or test data before deployment. With formulas, P(Xtraining) similar to P(XtoDeploy)
Data drift : Evolution of the production data over time compared to training or test data before deployment. With formulas, compare P(Xtraining) to P(XProduction)
Model drift : Model performances' evolution over time due to change in the target feature statistical properties (Concept drift), or due to change in data (Data drift). With formulas, when change in P(Y|XProduction) compared to P(Y|Xtraining) is concept drift. And change in P(Y,XProduction) compared to P(Y,Xtraining) is model drift

Eurybia helps data analysts and data scientists to collaborate through a report that allows them to exchange on drift monitoring and data validation before deploying model into production.
Eurybia also contributes to data science auditing by displaying usefull information about any model and data in a unique report.

- Readthedocs: [![documentation badge](https://readthedocs.org/projects/eurybia/badge/?version=latest)](https://eurybia.readthedocs.io/en/latest/)
- Medium:
- [Report Example](https://eurybia.readthedocs.io/en/latest/report.html)


## 🔥 Features

- Display clear and understandable insightful report :

<p align="center">
  <img align="left" src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_features_importance.PNG?raw=true" width="28%"/>
  <img src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_scatter_plot.PNG?raw=true" width="28%" />
  <img align="right" src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_auc_plot.PNG?raw=true" width="20%" />
</p>

<p align="center">
  <img align="left" src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_contribution_plot.PNG?raw=true" width="28%" />
  <img src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia-fond-clair.png?raw=true" width="15%" />
  <img align="right" src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_univariate_continuous.PNG?raw=true" width="28%" />
</p>

<p align="center">
  <img align="left" src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_contribution_violin.PNG?raw=true" width="28%" />
  <img src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_univariate_categorial.PNG?raw=true" width="28%" />
  <img align="right" src="https://github.com/MAIF/eurybia/blob/master/docs/_static/eurybia_auc_evolution.PNG?raw=true" width="28%" />
</p>


- Allow Data Scientists to quickly explore drift thanks to **dynamic reports** to easily navigate between drift detection and datasets features :

  <p align="center">
    <img src="https://github.com/MAIF/eurybia/blob/master/docs/_static/report_scrolling.gif?raw=true" width="800" title="contrib">
  </p>


**In a nutshell** :

- Monitoring drift using a scheduler (like Airflow)

- Evaluate level of data drift

- Facilitate collaboration between data analysts and data scientists, and easily share and discuss results with non-Data users

**More precisely** :
- **Render** data drift and model drift over time through :
    - Feature importance: features that discriminate the most the two datasets
    - Scatter plot: Feature importance relatively to the drift importance
    - Dataset analysis: distribution comparison between variable from the baseline dataset and the newest one
    - Predicted values analysis: distribution comparison between targets from the baseline dataset and the newest one
    - Performance of the data drift classifier
    - Features contribution for the data drift classifier
    - AUC evolution: comparison of data drift classifier at different period.
    - Model performance evolution: your model performances over time


## ⚙️ How Eurybia works

**Eurybia** works mainly with a binary classification model (named datadrift classifier) that tries to predict whether a sample belongs to the training dataset (or baseline dataset) or to the production dataset (or current dataset).

<p align="center">
  <img src="https://github.com/MAIF/eurybia/blob/master/docs/_static/data_drift_detection.png?raw=true" width="60%" />
</p>

As shown below on the diagram, there are 2 datasets, the baseline and the current one. Those datasets are those we wish to compare in order to assess if data drift occurred. On the first one we create a column named “target”, it will be filled only with 0, on the other hand on the second dataset we also add this column, but this time it will be filled only with 1 values.
Our goal is to build a binary classification model on top of those 2 datasets (concatenated). Once trained, this model will be helpful to tell if there is any data drift. To do so we are looking at the model performance through AUC metric. The greater the AUC the greater the drift is. (AUC = 0.5 means no data drift and AUC close to 1 means data drift is occuring)

The explainability of this datadrift classifier allows to prioritise features that are important for drift and to focus on those that have the most impact on the model in production.

To use Eurybia to monitor drift over time, you can use a scheduler to make computations automatically and periodically.
One of the schedulers you can use is Apache Airflow. To use it, you can read the [official documentation](https://airflow.apache.org/) and read blogs like this one: [Getting started with Apache Airflow](https://towardsdatascience.com/getting-started-with-apache-airflow-df1aa77d7b1b)


## 🛠 Installation

Eurybia is intended to work with Python versions 3.7 to 3.9. Installation can be done with pip:

```
pip install eurybia
```

If you encounter **compatibility issues** you may check the corresponding section in the Eurybia documentation [here](https://eurybia.readthedocs.io/en/latest/installation-instructions/index.html).
## 🕐 Quickstart

The 3 steps to display results:

- Step 1: Declare SmartDrift Object
  > you need to pass at least 2 pandas DataFrames in order to instantiate the SmartDrift class (Current or production dataset, baseline or training dataset)

```
from eurybia import SmartDrift
sd = SmartDrift(
  df_current=df_current,
  df_baseline=df_baseline,
  deployed_model=my_model, # Optional: put in perspective result with importance on deployed model
  encoding=my_encoder # Optional: if deployed_model and encoder to use this model
  )
```

- Step 2: Compile Model
  > There are different ways to compile the SmartDrift object

```
sd.compile(
  full_validation=True, # Optional: to save time, leave the default False value. If True, analyze consistency on modalities between columns.
  date_compile_auc='01/01/2022', # Optional: useful when computing the drift for a time that is not now
  datadrift_file="datadrift_auc.csv", # Optional: name of the csv file that contains the performance history of data drift
  )
```

- Step 3: Generate report
  > The report's content will be enriched if you provided the datascience model (deployed) and its encoder.
  Note that providing the deployed_model and encoding will only produce useful results if the datasets are both usable by the model (i.e. all features are present, dtypes are correct, etc).

```
sd.generate_report(
  output_file='output/my_report_name.html',
  title_story="my_report_title",
  title_description="my_report_subtitle", # Optional: add a subtitle to describe report
  project_info_file='project_info.yml' # Optional: add information on report
  )
```

[Report Example](https://eurybia.readthedocs.io/en/latest/report.html)


## 📖  Tutorials

This github repository offers a lot of tutorials to allow you to start more concretely in the use of Eurybia.

### Overview
- [Overview to compile Eurybia and generate Report](tutorial/tutorial01-Eurybia-overview.ipynb)

### Validate Data before model deployment
- [**Eurybia** Data Validation](tutorial/data_validation/tutorial01-data-validation.ipynb)
- [Validate data in production for model deployment](tutorial/data_validation/tutorial02-data-validation-iteration.ipynb)

### Measure and analyze Data drift
- [**Eurybia** to monitor Data Drift](tutorial/data_drift/tutorial01-datadrift-over-years.ipynb)
- [Detect high data drift over years](tutorial/data_drift/tutorial02-datadrift-high-datadrift.ipynb)

### Measure and analyze Model drift
- [**Eurybia** to detect Model Drift](tutorial/model_drift/tutorial01-modeldrift.ipynb)
- [Detect high model drift over years](tutorial/model_drift/tutorial02-modeldrift-high-datadrift.ipynb)

### More details about report and plots
- [Customize colors in report and plots](tutorial/common/tuto-common01-colors.ipynb)
- [Use **Shapash** Webapp to understand datadrift classifier](tutorial/common/tuto-common02-shapash-webapp.ipynb)
