"""
VMP 2023-04-10:
- plotting functions
- correlation functions
- overall metrics functions
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    matthews_corrcoef,
)
from itertools import combinations


### for plotting ###
def single_lineplot(
    df,
    metric,
    hue,
    figsize=(8, 5),
    alpha=1,
    ncol_legend=None,
    palette=None,
    hue_order=None,
    outpath=None,
    outname=None,
):
    # basic plot setup
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")

    # group by Percent and hue, average metric
    df_grouped = df.groupby(["Percent", hue])[metric].mean().reset_index(name="average")

    # line-plot
    sns.lineplot(
        data=df_grouped,
        x="Percent",
        y="average",
        hue=hue,
        hue_order=hue_order,
        alpha=alpha,
        marker="o",
        palette=palette,
    )

    # aesthetics
    plt.legend().remove()
    plt.ylabel(f"Average {metric}", fontsize=12)
    plt.xlabel("Percent Missing", fontsize=12)
    x_ticks = df["Percent"].unique()
    plt.xticks(x_ticks)
    plt.tight_layout()

    # add legend
    handles, labels = ax.get_legend_handles_labels()
    ncol = ncol_legend if ncol_legend else len(handles)
    nrow = np.ceil(len(df[hue].unique()) / ncol)
    y_align_legend = -0.08 * nrow
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y_align_legend),
        ncol=ncol,
        frameon=False,
        fontsize=12,
    )

    plt.subplots_adjust(left=0.15, bottom=0.1)

    # save or show
    if outpath:
        plt.savefig(os.path.join(outpath, outname), dpi=300, bbox_inches="tight")
    else:
        plt.show()


def multiple_lineplots(
    df,
    metric,
    hue,
    grid,
    color_dict=None,  # Allows custom color for each hue value
    legend=True,
    legend_order=None,  # Added legend_order parameter
    figsize=(7, 7),
    alpha=1,
    sharey="none",
    ncol_legend=None,
    outpath=None,
    outname=None,
):
    unique_grid_values = df[grid].unique()
    n_subplots = len(unique_grid_values)
    fig, ax = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    fig.patch.set_facecolor("white")

    # Initialize variables to determine global y-axis limits if needed
    global_ymin = float("inf")
    global_ymax = float("-inf")

    for num, ele in enumerate(unique_grid_values):
        # Select the subset of data for the current grid value
        df_subset = df[df[grid] == ele]

        # Use the provided legend order or default to unique hue values
        hue_values = legend_order if legend_order else df_subset[hue].unique()

        for hue_value in hue_values:
            # Subset the data for each hue value
            df_grouped = (
                df_subset[df_subset[hue] == hue_value]
                .groupby(["Percent"])[metric]
                .mean()
                .reset_index(name="average")
            )

            # Plotting with specified color from the color_dict or default color
            plot_color = (
                color_dict.get(hue_value)
                if color_dict and hue_value in color_dict
                else None
            )

            sns.lineplot(
                data=df_grouped,
                x="Percent",
                y="average",
                label=hue_value if legend else None,
                color=plot_color,
                alpha=alpha,
                marker="o",
                ax=ax[num],
            )

            # Update global y-axis limits
            ymin, ymax = df_grouped["average"].min(), df_grouped["average"].max()
            global_ymin, global_ymax = min(global_ymin, ymin), max(global_ymax, ymax)

        # Remove the legend from individual plots
        ax[num].legend().remove()
        ax[num].set_ylabel("")
        ax[num].set_title(f"{metric} ({grid} = {ele})")
        ax[num].set_xlabel("Percent Missing", fontsize=12)
        x_ticks = df_subset["Percent"].unique()
        ax[num].set_xticks(x_ticks)

    # If sharey='all', set the same y-axis limits for all subplots
    global_ymin = global_ymin - global_ymax * 0.05
    global_ymax = global_ymax + global_ymax * 0.05
    if sharey == "all":
        for a in ax:
            a.set_ylim(global_ymin, global_ymax)

    # Set a global y-axis label for the entire figure instead of each subplot
    fig.text(
        0.05, 0.5, "Average " + metric, va="center", rotation="vertical", fontsize=12
    )

    # Only add the legend if specified
    plt.tight_layout()
    if legend:
        handles, labels = ax[0].get_legend_handles_labels()

        # Reorder the legend handles and labels according to the provided legend_order
        if legend_order:
            # Create a dictionary mapping labels to handles
            handle_dict = dict(zip(labels, handles))
            handles = [
                handle_dict[label] for label in legend_order if label in handle_dict
            ]
            labels = [label for label in legend_order if label in handle_dict]

        ncol = ncol_legend if ncol_legend else len(handles)
        nrow = int(np.ceil(n_subplots / ncol))
        y_align_legend = -0.1 * nrow
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, y_align_legend),
            ncol=ncol,
            frameon=False,
            fontsize=12,
        )

    plt.subplots_adjust(left=0.15, bottom=0.1)

    # Save or show
    if outpath:
        plt.savefig(os.path.join(outpath, outname), dpi=300, bbox_inches="tight")
    else:
        plt.show()


#### for correlations ####
def calculate_correlation(df, column_x, column_y):
    """Calculate the correlation between two columns in a DataFrame."""
    df_filter = df[[column_x, column_y]]
    correlation_matrix = df_filter.corr().to_numpy()
    correlation_value = correlation_matrix[np.triu_indices(2, k=1)][0]
    return correlation_value


def flag_constant_variables(methods, df_complete, data_path):
    """if a variable is constant in any nan file, flag it as constant (for correlations)"""
    question_columns = [col for col in df_complete.columns if col.startswith("X")]
    constant_columns = []
    for method in methods:
        files = os.listdir(os.path.join(data_path, method))
        for file in files:
            df_impute = pd.read_csv(os.path.join(data_path, method, file))
            for question_column in question_columns:
                if len(df_impute[question_column].unique()) == 1:
                    constant_columns.append(question_column)
    constant_columns = list(np.unique(np.array(constant_columns)))
    return constant_columns


def process_file_study1(method, file, data_path, column_pairs, df_complete, pattern):
    """Process each file to compute correlations."""
    # print(pattern)
    # print(file)
    # print(data_path)
    _, type_, percent, iter_ = re.match(pattern, file).groups()
    # print(method)
    inpath_imputed = os.path.join(data_path, method, file)
    # print(inpath_imputed)
    imputed_data = pd.read_csv(inpath_imputed)
    evaluations = []
    for column_x, column_y in column_pairs:
        complete_value = calculate_correlation(df_complete, column_x, column_y)
        imputed_value = calculate_correlation(imputed_data, column_x, column_y)
        evaluations.append(
            [
                type_,
                percent,
                iter_,
                method,
                column_x,
                column_y,
                complete_value,
                imputed_value,
            ]
        )
    return evaluations


def gather_data_study1(data_list):
    evaluation_df = pd.DataFrame(
        data_list,
        columns=[
            "Type",
            "Percent",
            "Iter",
            "Method",
            "var_x",
            "var_y",
            "corr_complete",
            "corr_imputed",
        ],
    )
    evaluation_df["Iter"] = evaluation_df["Iter"].astype(int)
    evaluation_df = evaluation_df.sort_values(by=["Type", "Percent", "Iter"])
    return evaluation_df


def pairwise_correlations_study1(
    methods, df_complete, data_path, constant_variables=None
):
    question_columns = [col for col in df_complete.columns if col.startswith("X")]
    question_columns = [
        col for col in question_columns if col not in constant_variables
    ]
    column_pairs = list(combinations(question_columns, 2))

    pattern = r"^(.*?)_(MCAR|MAR|MNAR)_(\d+)_(\d+)\..*?(?:csv|txt)$"
    evaluation_list = []
    for method in methods:
        print(method)
        files = os.listdir(os.path.join(data_path, method))
        for file in files:
            evaluations = process_file_study1(
                method, file, data_path, column_pairs, df_complete, pattern
            )
            evaluation_list.extend(evaluations)

    evaluation_df = gather_data_study1(evaluation_list)
    return evaluation_df


### for overall metrics ###
def calculate_metrics(y_true, y_pred):

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


def process_column_study1(
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


def process_column_study2(
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
    # where we have nan in additional nan
    df_nan_c = df_nan[question_column]
    mat_nan_c = df_nan_c.to_numpy()
    mask_nan_c = np.isnan(mat_nan_c)
    # where we do not have nan in original data (otherwise we cannot compare)
    df_complete_c = df_complete[question_column]
    mat_complete_c = df_complete_c.to_numpy()
    mask_complete_c = ~np.isnan(mat_complete_c)
    # combine the masks (i.e., where we have added nan, but not nan in original)
    combined_mask_c = mask_nan_c & mask_complete_c
    # get imputed values
    df_impute_c = df_impute[question_column]
    mat_impute_c = df_impute_c.to_numpy()
    # get y_pred and y_true
    y_pred = mat_impute_c[combined_mask_c]
    y_true = mat_complete_c[combined_mask_c]
    # calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    # append data to list
    data_list.append(
        [question_column, method, missing_type, missing_percent, iter, *metrics]
    )
    return data_list


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


def basic_metrics_study1(method_grid, df_complete, nan_path, data_path):
    pattern = r"NA_(MCAR|MNAR|MAR)_(\d+)_(\d+).csv"
    question_columns = [col for col in df_complete.columns if col.startswith("X")]
    df_complete = df_complete[question_columns]
    nan_files = os.listdir(nan_path)
    data_list = []

    for nan_file in nan_files:
        missing_type, missing_percent, iter = re.match(pattern, nan_file).groups()
        df_nan = pd.read_csv(os.path.join(nan_path, nan_file))

        for method in method_grid:
            filename = f"{method}_{missing_type}_{missing_percent}_{iter}.csv"
            df_impute = pd.read_csv(os.path.join(data_path, method, filename))

            for question_column in question_columns:
                process_column_study1(
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


def basic_metrics_study2(method_grid, df_complete, nan_path, data_path):
    pattern = r"NA_(MCAR|MNAR|MAR)_(\d+)_(\d+).csv"
    question_columns = [col for col in df_complete.columns if col.startswith("X")]
    df_complete = df_complete[question_columns]
    nan_files = os.listdir(nan_path)
    data_list = []

    for nan_file in nan_files:
        missing_type, missing_percent, iter = re.match(pattern, nan_file).groups()
        df_nan = pd.read_csv(os.path.join(nan_path, nan_file))

        for method in method_grid:
            filename = f"{method}_{missing_type}_{missing_percent}_{iter}.csv"
            df_impute = pd.read_csv(os.path.join(data_path, method, filename))

            for question_column in question_columns:
                process_column_study2(
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


### for multiple imputation ###
def process_file_study3(file, data_path, column_pairs, df_complete, pattern):
    """Process each imputed file to compute correlations."""
    method, type_, percent, iter_, imputation_iter = re.match(pattern, file).groups()
    inpath_imputed = os.path.join(data_path, method, file)
    imputed_data = pd.read_csv(inpath_imputed)
    evaluations = []
    for column_x, column_y in column_pairs:
        complete_value = calculate_correlation(df_complete, column_x, column_y)
        imputed_value = calculate_correlation(imputed_data, column_x, column_y)
        evaluations.append(
            [
                type_,
                percent,
                iter_,
                imputation_iter,
                method,
                column_x,
                column_y,
                complete_value,
                imputed_value,
            ]
        )
    return evaluations


def gather_data_study3(data_list):
    evaluation_df = pd.DataFrame(
        data_list,
        columns=[
            "Type",
            "Percent",
            "Iter",
            "Imputation_Iter",
            "Method",
            "var_x",
            "var_y",
            "corr_complete",
            "corr_imputed",
        ],
    )
    evaluation_df[["Iter", "Imputation_Iter"]] = evaluation_df[
        ["Iter", "Imputation_Iter"]
    ].astype(int)
    evaluation_df = evaluation_df.sort_values(
        by=["Type", "Percent", "Iter", "Imputation_Iter"]
    )
    return evaluation_df


def pairwise_correlations_study3(
    methods, df_complete, data_path, constant_variables=None
):
    question_columns = [col for col in df_complete.columns if col.startswith("X")]
    question_columns = [
        col for col in question_columns if col not in constant_variables
    ]
    column_pairs = list(combinations(question_columns, 2))

    # Updated pattern to include imputation iteration
    pattern = r"^(.*?)_(MCAR|MAR|MNAR)_(\d+)_(\d+)_(\d+)\..*?(?:csv|txt)$"
    evaluation_list = []

    for method in methods:
        files = os.listdir(os.path.join(data_path, method))
        for file in files:
            evaluations = process_file_study3(
                file, data_path, column_pairs, df_complete, pattern
            )
            evaluation_list.extend(evaluations)

    evaluation_df = gather_data_study3(evaluation_list)
    return evaluation_df
