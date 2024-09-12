import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # this makes sense here for "precision" warning
# from evaluation_functions import (
#    basic_metrics_study1,
#    flag_constant_variables,
# )
# from constants import method_grid

# paths to nan files
nan_path = "../data/study1/additional_NA"
data_path = "../data/study1/"
df_complete = pd.read_csv("../data/preprocessed/answers_study1.csv")


import re
import os
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


def calculate_metrics(y_true, y_pred):
    # for MAR we have some columns with no missing values
    # this is basically the point of MAR.
    if y_pred.shape[0] == 0:
        mpb = np.nan
        rmse = np.nan
        accuracy = np.nan
        precision = np.nan
        recall = np.nan
        f1 = np.nan
        matthews_corr = np.nan

    else:
        # Calculate Mean Percent Bias
        # NB: not quite MPB but this makes more sense
        mpb = np.abs(np.mean((y_true - y_pred) / 1)) * 100

        # inverse accuracy for just mse because 0/1
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # undefined (or meaningless) if all values are 0 or 1
        if np.all(y_true == 0) or np.all(y_true == 1):
            matthews_corr = np.nan
        else:
            matthews_corr = matthews_corrcoef(y_true, y_pred)

        # undefined (or meaningless) if all values are 0
        # consider whether this should also check all true == 1
        if np.all(y_true == 0):
            precision = np.nan
            recall = np.nan
            f1 = np.nan
        else:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

    return rmse, mpb, accuracy, precision, recall, f1, matthews_corr


def variable_metrics(
    df_complete,
    df_nan,
    df_impute,
    question_column,
    method,
    missing_type,
    missing_percent,
    iter,
    data_list,
):
    # get y_true and y_pred for specific indices (where we have nan in additional nan)
    valid_indices = df_nan[question_column].isna().to_numpy()
    y_true = df_complete[question_column].to_numpy()[valid_indices]
    y_pred = df_impute[question_column].to_numpy()[valid_indices]

    # calculate metrics
    metrics = calculate_metrics(y_true, y_pred)

    # append data to list
    data_list.append(
        [question_column, method, missing_type, missing_percent, iter, *metrics]
    )


def gather_metrics(data_list):
    df_metrics = pd.DataFrame(
        data_list,
        columns=[
            "Question",
            "Method",
            "Type",
            "Percent",
            "Iter",
            "RMSE",
            "Mean Percent Bias",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 score",
            "Matthews Correlation",
        ],
    )
    df_metrics["Iter"] = df_metrics["Iter"].astype(int)
    df_metrics = df_metrics.sort_values(by=["Method", "Type", "Percent", "Iter"])
    return df_metrics


def metrics_study1(method_grid, df_complete, nan_path, data_path):
    pattern = r"NA_(MCAR|MNAR|MAR)_(\d+)_(\d+).csv"
    question_columns = [col for col in df_complete.columns if col.startswith("Q_")]
    df_complete = df_complete[question_columns]
    nan_files = os.listdir(nan_path)
    data_list = []

    for nan_file in nan_files:
        missing_type, missing_percent, iter = re.match(pattern, nan_file).groups()
        df_nan = pd.read_csv(os.path.join(nan_path, nan_file))
        for method in method_grid:
            df_impute = pd.read_csv(os.path.join(data_path, method, nan_file))

            for question_column in question_columns:
                variable_metrics(
                    df_complete,
                    df_nan,
                    df_impute,
                    question_column,
                    method,
                    missing_type,
                    missing_percent,
                    iter,
                    data_list,
                )

    df_metrics = gather_metrics(data_list)
    return df_metrics


### calculate basic metrics ###
method_grid = ["mode", "knn", "ridge", "rf", "xgb"]
df_metrics = metrics_study1(method_grid, df_complete, nan_path, data_path)

df_metrics.to_csv("../data/evaluation/metrics_study1.csv", index=False)
