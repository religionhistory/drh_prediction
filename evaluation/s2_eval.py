import numpy as np
import pandas as pd
from evaluation_functions import (
    remove_all_nan,
    fill_zero,
    remove_any_nan,
    multiple_lineplots,
)

study = "study2"

# load metrics files for each method
metrics_knn = pd.read_csv(f"../data/{study}/evaluation/metrics_knn.csv")
metrics_rf = pd.read_csv(f"../data/{study}/evaluation/metrics_rf.csv")
metrics_ridge = pd.read_csv(f"../data/{study}/evaluation/metrics_ridge.csv")
metrics_xgb = pd.read_csv(f"../data/{study}/evaluation/metrics_xgb.csv")
metrics_mode = pd.read_csv(f"../data/{study}/evaluation/metrics_mode.csv")

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

## aggregate on questions
df_removenan_agg = (
    df_removenan.groupby(
        ["missing_type", "missing_percent", "metric", "method", "iteration"]
    )["value"]
    .mean()
    .reset_index()
)
df_fillzero_agg = (
    df_fillzero.groupby(
        ["missing_type", "missing_percent", "metric", "method", "iteration"]
    )["value"]
    .mean()
    .reset_index()
)


"""
Generate tables.
No meaningful differences between 10-20% additional missing data.
So just aggregate everything into one measure. 
"""

# removenan overall table
df_removenan_table = (
    df_removenan_agg.groupby(["metric", "method"])["value"].mean().unstack()
)
df_removenan_table = df_removenan_table.round(3)
df_removenan_table.to_latex("../tables/s2_removenan_table.tex", float_format="%.3f")

# fillzero overall table
df_fillzero_table = (
    df_fillzero_agg.groupby(["metric", "method"])["value"].mean().unstack()
)
df_fillzero_table = df_fillzero_table.round(3)
df_fillzero_table.to_latex("../tables/s2_fillzero_table.tex", float_format="%.3f")

# show by question level
## get question relations
question_relations = pd.read_csv(f"../data/preprocessed/question_level_{study}.csv")
question_relations["question_id"] = "Q_" + question_relations["question_id"].astype(str)
question_relations = question_relations[["question_id", "question_level"]]
question_relations = question_relations.rename(columns={"question_id": "column"})

# merge question relations
df_removenan_questions = df_removenan.merge(
    question_relations, on="column", how="inner"
)
df_removenan_questions_f1 = df_removenan_questions[
    df_removenan_questions["metric"] == "f1_score"
]

df_qlevel_table = (
    df_removenan_questions_f1.groupby(["question_level", "method"])["value"]
    .mean()
    .unstack()
)
df_qlevel_table = df_qlevel_table.round(3)
df_qlevel_table.to_latex("../tables/s2_qlevel_table.tex", float_format="%.3f")
