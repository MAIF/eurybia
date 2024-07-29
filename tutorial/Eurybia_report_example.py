"""
This script can be used to generate the report example.
For more information, please refer to the tutorial 'tutorial01-modeldrift.ipynb' in tutorial/model_drift
that generates the same report.
"""
import os
import sys
from pathlib import Path

import catboost
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../..")

from eurybia import SmartDrift
from eurybia.data.data_loader import data_loading

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    df_car_accident = data_loading("us_car_accident")
    df_accident_baseline = df_car_accident.loc[df_car_accident["year_acc"] == 2016]
    df_accident_2017 = df_car_accident.loc[df_car_accident["year_acc"] == 2017]
    df_accident_2018 = df_car_accident.loc[df_car_accident["year_acc"] == 2018]
    df_accident_2019 = df_car_accident.loc[df_car_accident["year_acc"] == 2019]
    df_accident_2020 = df_car_accident.loc[df_car_accident["year_acc"] == 2020]
    df_accident_2021 = df_car_accident.loc[df_car_accident["year_acc"] == 2021]

    y_df_learning = df_accident_baseline["target"].to_frame()
    X_df_learning = df_accident_baseline[
        df_accident_baseline.columns.difference(["target", "target_multi", "year_acc", "Description"])
    ]

    y_df_2017 = df_accident_2017["target"].to_frame()
    X_df_2017 = df_accident_2017[
        df_accident_2017.columns.difference(["target", "target_multi", "year_acc", "Description"])
    ]

    y_df_2018 = df_accident_2018["target"].to_frame()
    X_df_2018 = df_accident_2018[
        df_accident_2018.columns.difference(["target", "target_multi", "year_acc", "Description"])
    ]

    y_df_2019 = df_accident_2019["target"].to_frame()
    X_df_2019 = df_accident_2019[
        df_accident_2019.columns.difference(["target", "target_multi", "year_acc", "Description"])
    ]

    y_df_2020 = df_accident_2020["target"].to_frame()
    X_df_2020 = df_accident_2020[
        df_accident_2020.columns.difference(["target", "target_multi", "year_acc", "Description"])
    ]

    y_df_2021 = df_accident_2021["target"].to_frame()
    X_df_2021 = df_accident_2021[
        df_accident_2021.columns.difference(["target", "target_multi", "year_acc", "Description"])
    ]

    features = [
        "Start_Lat",
        "Start_Lng",
        "Distance(mi)",
        "Temperature(F)",
        "Humidity(%)",
        "Visibility(mi)",
        "day_of_week_acc",
        "Nautical_Twilight",
        "season_acc",
    ]

    features_to_encode = [
        col for col in X_df_learning[features].columns if X_df_learning[col].dtype not in ("float64", "int64")
    ]
    encoder = OrdinalEncoder(cols=features_to_encode)
    encoder = encoder.fit(X_df_learning[features])
    X_df_learning_encoded = encoder.transform(X_df_learning)

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X_df_learning_encoded, y_df_learning, train_size=0.75, random_state=1
    )
    train_pool_cat = catboost.Pool(data=Xtrain, label=ytrain, cat_features=features_to_encode)
    test_pool_cat = catboost.Pool(data=Xtest, label=ytest, cat_features=features_to_encode)

    model = catboost.CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        learning_rate=0.143852,
        iterations=300,
        l2_leaf_reg=15,
        max_depth=4,
        use_best_model=True,
        custom_loss=["Accuracy", "AUC", "Logloss"],
    )

    model = model.fit(train_pool_cat, plot=False, eval_set=test_pool_cat, verbose=False)
    SD = SmartDrift(df_current=X_df_2017, df_baseline=X_df_learning, deployed_model=model, encoding=encoder)
    SD.compile(
        full_validation=True,
        date_compile_auc="01/01/2017",
        datadrift_file=os.path.join(cur_dir, "car_accident_auc.csv"),
    )

    proba = model.predict_proba(X_df_2017)
    performance = metrics.roc_auc_score(y_df_2017, proba[:, 1]).round(5)
    df_performance = pd.DataFrame({"annee": [2017], "mois": [1], "performance": [performance]})

    SD = SmartDrift(df_current=X_df_2018, df_baseline=X_df_learning, deployed_model=model, encoding=encoder)
    SD.compile(
        full_validation=True,
        date_compile_auc="01/01/2018",  # optionnal, by default date of compile
        datadrift_file=os.path.join(cur_dir, "car_accident_auc.csv"),
    )

    proba = model.predict_proba(X_df_2018)
    performance = metrics.roc_auc_score(y_df_2018, proba[:, 1]).round(5)
    df_performance = df_performance.append({"annee": 2018, "mois": 1, "performance": performance}, ignore_index=True)

    SD = SmartDrift(df_current=X_df_2019, df_baseline=X_df_learning, deployed_model=model, encoding=encoder)
    SD.compile(
        full_validation=True,
        date_compile_auc="01/01/2019",  # optionnal, by default date of compile
        datadrift_file=os.path.join(cur_dir, "car_accident_auc.csv"),
    )

    proba = model.predict_proba(X_df_2019)
    performance = metrics.roc_auc_score(y_df_2019, proba[:, 1]).round(5)
    df_performance = df_performance.append({"annee": 2019, "mois": 1, "performance": performance}, ignore_index=True)

    SD = SmartDrift(df_current=X_df_2020, df_baseline=X_df_learning, deployed_model=model, encoding=encoder)
    SD.compile(
        full_validation=True,
        date_compile_auc="01/01/2020",  # optionnal, by default date of compile
        datadrift_file=os.path.join(cur_dir, "car_accident_auc.csv"),
    )

    proba = model.predict_proba(X_df_2020)
    performance = metrics.roc_auc_score(y_df_2020, proba[:, 1]).round(5)
    df_performance = df_performance.append({"annee": 2020, "mois": 1, "performance": performance}, ignore_index=True)

    SD = SmartDrift(df_current=X_df_2021, df_baseline=X_df_learning, deployed_model=model, encoding=encoder)
    SD.compile(
        full_validation=True,
        date_compile_auc="01/01/2021",  # optionnal, by default date of compile
        datadrift_file=os.path.join(cur_dir, "car_accident_auc.csv"),
    )

    proba = model.predict_proba(X_df_2021)
    performance = metrics.roc_auc_score(y_df_2021, proba[:, 1]).round(5)
    df_performance = df_performance.append({"annee": 2021, "mois": 1, "performance": performance}, ignore_index=True)
    SD.add_data_modeldrift(dataset=df_performance, metric="performance")

    SD.generate_report(
        output_file=os.path.join(cur_dir, "report.html"),
        title_story="Model drift Report",
        title_description="""US Car accident model drift 2021""",
        project_info_file=os.path.join(
            Path(os.path.abspath(__file__)).parent.parent, "eurybia/data", "project_info_car_accident.yml"
        ),
    )
    os.remove(os.path.join(cur_dir, "car_accident_auc.csv"))
