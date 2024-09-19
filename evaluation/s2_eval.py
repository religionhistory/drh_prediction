import numpy as np
import pandas as pd
from evaluation_functions import (
    remove_all_nan,
    fill_zero,
    remove_any_nan,
    multiple_lineplots,
)

# load metrics files for each method
metrics_knn = pd.read_csv("../data/study2/evaluation/metrics_knn.csv")
metrics_rf = pd.read_csv("../data/study2/evaluation/metrics_rf.csv")
metrics_ridge = pd.read_csv("../data/study2/evaluation/metrics_ridge.csv")
metrics_xgb = pd.read_csv("../data/study2/evaluation/metrics_xgb.csv")
metrics_mode = pd.read_csv("../data/study2/evaluation/metrics_mode.csv")

# combine metrics
metrics_combined = pd.concat(
    [metrics_knn, metrics_rf, metrics_ridge, metrics_xgb, metrics_mode]
)

# melt to long format
metrics_combined = pd.melt(
    metrics_combined,
    id_vars=["missing_type", "missing_percent", "iteration", "column", "method"],
    value_vars=["accuracy", "f1_score", "mcc"],
    var_name="metric",
    value_name="value",
)

""" Step 1:
remove places where all methods have NaN
these are caused by some questions being very skewed
and then introducing NaN resulting in some datasets
with only 1 class.
"""

group_cols = ["missing_type", "missing_percent", "iteration", "column"]
df_clean = remove_all_nan(group_cols, metrics_combined)

""" Step 2: 
take out accuracy because this can be computed also
when only 1 class is predicted by the method
(whereas this does not make sense for F1 and MCC).
"""

## step 2 is only aimed at MCC and F1 score ##
# so we save accuracy for now with no NANs
df_accuracy = df_clean[df_clean["metric"] == "accuracy"]
assert df_accuracy.isna().sum().sum() == 0

# take out accuracy and mode
# mode is not meaningful for MCC and F1 score
# it will just by definition be set to zero.
df_metrics = df_clean[df_clean["metric"] != "accuracy"]
df_metrics = df_metrics[df_metrics["method"] != "mode"]

""" Step 3:
Fill missing F1 and MCC with 0.
This penalizes methods that predict only 1 class.
"""

df_fillzero = fill_zero(df_metrics)

""" Step 4:
Instead of filling with zero, we can also remove
the columns where ANY method has a NaN. 
This still only compares cases across methods that make sense
but does not penalize predicting only 1 class in some instances.
Note that we have high skew so the above is very strict. 
"""

df_removenan = remove_any_nan(group_cols, df_metrics)


""" Step 5:
Now we can create 2 datasets that we can compare. 
"""

## now we can create 2 dataframes ##
df_removenan = pd.concat([df_accuracy, df_removenan])
df_fillzero = pd.concat([df_accuracy, df_fillzero])

""" Step 6: 
Plot results 
"""

color_dict = {
    "mode": "tab:gray",
    "knn": "tab:blue",
    "ridge": "tab:orange",
    "rf": "tab:red",
    "xgb": "tab:green",
}

legend_order = ["mode", "knn", "ridge", "rf", "xgb"]

label_dict = {
    "accuracy": "Accuracy",
    "f1_score": "F1 Score",
    "mcc": "MCC",
}

outpath = "../figures/study2"


# main plot
multiple_lineplots(
    df=df_removenan,
    metric="value",
    hue="method",
    grid="metric",
    color_dict=color_dict,
    label_dict=label_dict,
    legend_order=legend_order,
    ncol_legend=5,
    y_tick_decimals=2,
    outpath=outpath,
    outname="metrics_removenan.png",
)

# for supplementary
multiple_lineplots(
    df=df_fillzero,
    metric="value",
    hue="method",
    grid="metric",
    color_dict=color_dict,
    label_dict=label_dict,
    legend_order=legend_order,
    ncol_legend=5,
    y_tick_decimals=2,
    outpath=outpath,
    outname="metrics_fillzero.png",
)

# for supplementary
multiple_lineplots(
    df=df_removenan[df_removenan["metric"] == "f1_score"],
    metric="value",
    hue="method",
    grid="missing_type",
    color_dict=color_dict,
    legend_order=legend_order,
    ncol_legend=5,
    sharey="all",
    y_tick_decimals=1,
    outpath=outpath,
    outname="f1_missingtype_removenan.png",
)

# for supplementary
multiple_lineplots(
    df=df_fillzero[df_fillzero["metric"] == "f1_score"],
    metric="value",
    hue="method",
    grid="missing_type",
    color_dict=color_dict,
    legend_order=legend_order,
    ncol_legend=5,
    sharey="all",
    y_tick_decimals=1,
    outpath=outpath,
    outname="f1_missingtype_fillzero.png",
)

# could just show an overall table
# no large differences between the methods so that might be enough
# save this to latex
df_removenan.groupby(["metric", "method"])["value"].mean().unstack()

# this would also be a great table to show in the paper
# save this to latex
question_relations = pd.read_csv("../data/preprocessed/question_level_study2.csv")
question_relations["question_id"] = "Q_" + question_relations["question_id"].astype(str)
question_relations = question_relations[["question_id", "question_level"]]
question_relations = question_relations.rename(columns={"question_id": "column"})
df_removenan = df_removenan.merge(question_relations, on="column", how="inner")
df_removenan = df_removenan[df_removenan["metric"] == "f1_score"]
df_removenan.groupby(["question_level", "method"])["value"].mean().unstack()
