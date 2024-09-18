import numpy as np
import pandas as pd

metrics_knn = pd.read_csv("../data/study1/evaluation/metrics_knn.csv")
metrics_rf = pd.read_csv("../data/study1/evaluation/metrics_rf.csv")
metrics_ridge = pd.read_csv("../data/study1/evaluation/metrics_ridge.csv")
metrics_xgb = pd.read_csv("../data/study1/evaluation/metrics_xgb.csv")
metrics_mode = pd.read_csv("../data/study1/evaluation/metrics_mode.csv")

# combine metrics
metrics_combined = pd.concat(
    [metrics_knn, metrics_rf, metrics_ridge, metrics_xgb, metrics_mode]
)

# change format to long
metrics_combined = pd.melt(
    metrics_combined,
    id_vars=["missing_type", "missing_percent", "iter", "column", "method"],
    value_vars=["accuracy", "f1_score", "mcc"],
    var_name="metric",
    value_name="value",
)

# Step 1: Identify combinations where all methods have NaN for all metrics
group_cols = ["missing_type", "missing_percent", "iter", "column"]

# Create a flag indicating whether the value is NaN
metrics_combined["is_nan"] = metrics_combined["value"].isna()

# Group by the specified columns and check if all methods have NaN for all metrics
nan_groups = metrics_combined.groupby(group_cols)["is_nan"].all().reset_index()

# Identify combinations where is_nan is True (all values are NaN)
nan_combinations = nan_groups[nan_groups["is_nan"] == True][group_cols]

# Merge to filter out these combinations from the original DataFrame
df_clean = metrics_combined.merge(
    nan_combinations, on=group_cols, how="left", indicator=True
)
df_clean = df_clean[df_clean["_merge"] == "left_only"].drop(
    columns=["is_nan", "_merge"]
)

## step 2 is only aimed at MCC and F1 score ##
# so we save accuracy for now with no NANs
df_accuracy = df_clean[df_clean["metric"] == "accuracy"]
assert df_accuracy.isna().sum().sum() == 0

# take out accuracy and mode
# mode is not meaningful for MCC and F1 score
# it will just by definition be set to zero.
df_metrics = df_clean[df_clean["metric"] != "accuracy"]
df_metrics = df_metrics[df_metrics["method"] != "mode"]

## now we first fill with zeros ##
df_fillzero = df_metrics.copy()
df_fillzero["value"] = df_fillzero["value"].fillna(0.0)
assert df_fillzero.isna().sum().sum() == 0

## now we remove NAN ##
df_removenan = df_metrics.copy()
nan_any_combination = df_removenan[df_removenan["value"].isna()][
    group_cols
].drop_duplicates()
len(nan_any_combination)  # n = 49

df_removenan = df_removenan.merge(
    nan_any_combination, on=group_cols, how="left", indicator=True
)
df_removenan = df_removenan[df_removenan["_merge"] == "left_only"].drop(
    columns=["_merge"]
)

## now we can create 2 dataframes ##
df_removenan = pd.concat([df_accuracy, df_removenan])
df_fillzero = pd.concat([df_accuracy, df_fillzero])


# could do some stuff on individual questions
# which ones are predicted well and poorly, etc.
# but let us aggregate first.
def aggregate_columns(df, group_cols, value_col):
    df_agg = df.groupby(group_cols)[value_col].mean().reset_index()
    return df_agg


group_cols = ["method", "metric", "missing_type", "missing_percent", "iter"]
df_removenan_agg = aggregate_columns(df_removenan, group_cols, "value")
df_fillzero_agg = aggregate_columns(df_fillzero, group_cols, "value")

# just quick check for now
df_removenan_f1 = df_removenan[df_removenan["metric"] == "f1_score"]
df_removenan_f1.groupby(["method", "missing_percent"])["value"].mean()

df_fillzero_f1 = df_fillzero[df_fillzero["metric"] == "f1_score"]
df_fillzero_f1.groupby(["method", "missing_percent"])["value"].mean()
