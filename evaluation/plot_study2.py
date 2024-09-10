import numpy as np
import pandas as pd
from evaluation_functions import multiple_lineplots

# setup
outpath = "../figures/study2"
df_metrics = pd.read_csv("evaluation/metrics_study2.csv")
df_long = pd.melt(
    df_metrics,
    id_vars=["Question", "Method", "Type", "Percent", "Iter"],
    value_vars=["Accuracy", "Mean Percent Bias", "Matthews Correlation"],
    var_name="metric",
    value_name="values",
)

# overall metrics plot #
color_dict = {
    "mode": "tab:gray",
    "mice": "tab:blue",
    "miceSample": "tab:purple",
    "micePMM": "tab:green",
    "miceCART": "tab:brown",
    "miceRF": "tab:red",
    "missForest": "tab:orange",
}
legend_order = [
    "mode",
    "mice",
    "miceSample",
    "micePMM",
    "miceCART",
    "miceRF",
    "missForest",
]
multiple_lineplots(
    df=df_long,
    metric="values",
    hue="Method",
    grid="metric",
    ncol_legend=4,
    color_dict=color_dict,
    legend_order=legend_order,
    outpath=outpath,
    outname="overall_metrics.png",
)

# plot of average error on sub-questions and super-questions #
question_level = pd.read_csv("../data/preprocessed/question_level_study2.csv")


def format_question_level(df):
    df = df.rename({"question_id": "Question"}, axis=1)
    df["Question"] = df["Question"].astype(str)
    df["Question"] = "X" + df["Question"]
    return df


question_level_formatted = format_question_level(question_level)

# merge and aggregate
df_long_level = df_long.merge(question_level_formatted, on="Question")
df_long_level_mf = df_long_level[df_long_level["Method"] == "missForest"]
df_long_level_mf = (
    df_long_level_mf.groupby(["Type", "Percent", "Iter", "metric", "question_level"])[
        "values"
    ]
    .mean()
    .reset_index(name="values")
)

multiple_lineplots(
    df=df_long_level_mf,
    metric="values",
    hue="question_level",
    grid="metric",
    ncol_legend=4,
    outpath=outpath,
    outname="question_level.png",
)

# same plot but only for 1st and 2nd level #
level_1_questions = question_level[question_level["question_level"] == 1][
    "question_id"
].unique()
child_questions = question_level[
    question_level["parent_question_id"].isin(level_1_questions)
]["question_id"].unique()
parent_questions = question_level[question_level["question_id"].isin(child_questions)][
    "parent_question_id"
].unique()
valid_questions = np.concatenate((child_questions, parent_questions))
child_parent_pairs = question_level[question_level["question_id"].isin(valid_questions)]
question_level_formatted = format_question_level(child_parent_pairs)

# merge and aggregate
df_long_level = df_long.merge(question_level_formatted, on="Question")
df_long_level_mf = df_long_level[df_long_level["Method"] == "missForest"]
df_long_level_mf_agg = (
    df_long_level_mf.groupby(["Type", "Percent", "Iter", "metric", "question_level"])[
        "values"
    ]
    .mean()
    .reset_index(name="values")
)

multiple_lineplots(
    df=df_long_level_mf_agg,
    metric="values",
    hue="question_level",
    grid="metric",
    ncol_legend=4,
    outpath=outpath,
    outname="question_level_has_child.png",
)

# find out why it is so much worse:
# make a plot where we have a grouping but we do not average over questions
# so we only have two colors, but many lines ...
df_questions = pd.read_csv("../data/preprocessed/question_overview.csv")
df_questions["question_id"] = df_questions["question_id"].astype(str)
df_questions["question_id"] = "X" + df_questions["question_id"]
df_questions = df_questions.rename(columns={"question_id": "Question"})

df_long_level_mf_names = df_long_level_mf.merge(
    df_questions, on="Question", how="inner"
)

# check a case #
df_long_level_mf_sub = df_long_level_mf_names[
    (df_long_level_mf_names["Percent"] == 10)
    & (df_long_level_mf_names["metric"] == "Accuracy")
    & (df_long_level_mf_names["Type"] == "MAR")
]

df_long_level_mf_sub_agg = (
    df_long_level_mf_sub.groupby(["Question", "question_name", "parent_question_id"])[
        "values"
    ]
    .mean()
    .reset_index(name="mean_accuracy")
)

# do not truncate
pd.set_option("display.max_colwidth", None)
df_long_level_mf_sub_agg.head(5)

# check mean values for parents and child questions: #
answers_study_2 = pd.read_csv("../data/preprocessed/answers_study2.csv")
columns_starting_with_x = answers_study_2.filter(regex="^X").columns
filtered_df = answers_study_2[columns_starting_with_x]
mean_values = filtered_df.mean().reset_index(name="mean")
mean_values = mean_values.rename(columns={"index": "Question"})
df_long_mean = df_long_level_mf_sub_agg.merge(mean_values, on="Question", how="inner")

# good case study might be formal burials present (in cemetery).
# here the sub-question is more skewed than the super question.
# but we are still doing worse for the sub-question.
# think about the cases that we have here.

# load a data-set here #
columns = [
    "X4821",  # formal burials present
    "X4823",  # in cemetery
]
iter = 2
df_nan = pd.read_csv(f"../imputation/output/study2/additional_NA/NA_MAR_10_{iter}.csv")
df_imputed = pd.read_csv(
    f"../imputation/output/study2/missForest/missForest_MAR_10_{iter}.csv"
)
df_complete = pd.read_csv("../data/preprocessed/answers_study2.csv")


def process_column_study2(
    df_complete,
    df_nan,
    df_impute,
    question_column,
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
    # df_impute_c = df_impute[question_column]
    # mat_impute_c = df_impute_c.to_numpy()
    # get y_pred and y_true
    # y_pred = mat_impute_c[combined_mask_c]
    # y_true = mat_complete_c[combined_mask_c]
    # calculate metrics
    return combined_mask_c  # y_pred, y_true


combined_mask_first = process_column_study2(df_complete, df_nan, df_imputed, columns[0])
combined_mask_second = process_column_study2(
    df_complete, df_nan, df_imputed, columns[1]
)
combined_mask_both = combined_mask_first & combined_mask_second
df_imputed_combined = df_imputed[combined_mask_both][columns]
df_complete_combined = df_complete[combined_mask_both][columns]
df_complete_combined = df_complete_combined.rename(
    columns={"X4821": "X4821_complete", "X4823": "X4823_complete"}
)
df_concat = pd.concat([df_imputed_combined, df_complete_combined], axis=1)
df_concat["X4821_complete"] = df_concat["X4821_complete"].astype(int)
df_concat["X4823_complete"] = df_concat["X4823_complete"].astype(int)
df_concat["X4821_true"] = df_concat["X4821"] == df_concat["X4821_complete"]
df_concat["X4823_true"] = df_concat["X4823"] == df_concat["X4823_complete"]

df_nan["X4823"].mean()
df_nan["X4821"].mean()

# yeah so we are making some automatic errors by saying that
# no becomes no; but we are also making some automatic correct classifications.
# so still does not really explain what we are seeing.

y_pred_4821, y_true_4821 = process_column_study2(
    df_complete, df_nan, df_imputed, columns[0]
)
y_pred_4823, y_true_4823 = process_column_study2(
    df_complete, df_nan, df_imputed, columns[1]
)

np.sum(y_pred_4821 == y_true_4821) / len(y_pred_4821)  # 85
np.sum(y_pred_4823 == y_true_4823) / len(y_pred_4823)  # 77


""" 
y_pred 4823 can be shorter because there are more cases in the actual
data where we do not have observations for this question.
So even when we add more missingness, there are fewer comparisons that we can make. 
"""

# so the interesting thing to compute is cases
# where we have actually observed values for both
# and there have been added nan values for at least one.

# dataframe original #
question_column = "X4821"
# where we have nan in additional nan
df_nan_c = df_nan[question_column]
mat_nan_c = df_nan_c.to_numpy()
mask_nan_c = np.isnan(mat_nan_c)
sum(mask_nan_c)  # 159, 241
# where we do not have nan in original data (otherwise we cannot compare)
df_complete_c = df_complete[question_column]
mat_complete_c = df_complete_c.to_numpy()
mask_complete_c = ~np.isnan(mat_complete_c)
sum(mask_complete_c)  # 586, 405
# combine the masks (i.e., where we have added nan, but not nan in original)
combined_mask_c = mask_nan_c & mask_complete_c
sum(combined_mask_c)  # 37, 35
# get imputed values
df_impute_c = df_impute[question_column]
mat_impute_c = df_impute_c.to_numpy()
# get y_pred and y_true
y_pred = mat_impute_c[combined_mask_c]
y_true = mat_complete_c[combined_mask_c]
