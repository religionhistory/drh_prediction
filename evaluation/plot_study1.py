import pandas as pd
from evaluation_functions import multiple_lineplots, single_lineplot

# setup
outpath = "../figures/study1"
df_metrics = pd.read_csv("../data/evaluation/metrics_study1.csv")
df_long = pd.melt(
    df_metrics,
    id_vars=["Question", "Method", "Type", "Percent", "Iter"],
    value_vars=["Accuracy", "F1 score", "Matthews Correlation"],
    var_name="metric",
    value_name="values",
)
df_long = df_long.sort_values("Method")

# overall metrics plot #
color_dict = {
    "mode": "tab:gray",
    "knn": "tab:blue",
    "ridge": "tab:orange",
    "rf": "tab:red",
    "xgb": "tab:green",
}

legend_order = ["mode", "knn", "ridge", "rf", "xgb"]
multiple_lineplots(
    df=df_long,
    metric="values",
    hue="Method",
    grid="metric",
    color_dict=color_dict,
    legend_order=legend_order,
    ncol_legend=3,
    # outpath=outpath,
    # outname="overall_metrics.png",
)

""" 
Random Forest does best currently. 
Need to tweak hyper params for XGBoost & KNN. 
Also need to figure out which implementation of XGB. 
"""

# other metrics by missingness mechanism (e.g., MPB)
# the whole missingness mechanism I think is not so much
# a problem for prediction per se, but for imputation
# (i.e., other considerations than accuracy).
df_f1 = df_long[df_long["metric"] == "F1 score"]
multiple_lineplots(
    df=df_f1,
    grid="Type",
    metric="values",
    hue="Method",
    ncol_legend=4,
    color_dict=color_dict,
    # legend_order=hue_order,
    # outpath=outpath,
    # outname="MPB_missingness.png",
)

### prediction by question (for missForest) ###
df_long_missforest = df_long[df_long["Method"] == "rf"]
df_long_missforest["Question"].sort_values().unique()

# map questions
df_questions = pd.read_csv("../data/preprocessed/question_overview.csv")
from constants import question_mapping_study1

df_question_names = pd.DataFrame(
    question_mapping_study1.items(), columns=["question_id", "question_name_short"]
)
df_questions = df_questions.merge(df_question_names, on="question_id", how="inner")
df_questions["question_id"] = df_questions["question_id"].astype(str)
df_questions["question_id"] = "X" + df_questions["question_id"]
df_questions = df_questions.rename(columns={"question_id": "Question"})

# merge with missforest and select questions to display
df_long_missforest = df_long_missforest.merge(df_questions, on="Question", how="inner")
df_long_missforest[["Question", "question_name_short"]].sort_values(
    "Question"
).drop_duplicates()


selected_questions = [
    "Spirit-body distinction",
    "Afterlife belief",
    "Supernatural beings",
    "Supreme high god",
    "Supernatural monitoring",
    "Sacrifice children",
    "Written language",
    "Scriptures",
]

df_long_missforest = df_long_missforest[
    df_long_missforest["question_name_short"].isin(selected_questions)
]

multiple_lineplots(
    df=df_long_missforest,
    metric="values",
    hue="question_name_short",
    grid="metric",
    ncol_legend=2,
    outpath=outpath,
    outname="questions_missforest.png",
)

""" 
Some are almost always yes (supernatural beings) giving:
- no bias
- perfect accuracy
- worst possible MCC (consider just not including these for MCC).
"""

# understanding why some are great and some are bad.
answers_study1 = pd.read_csv("../data/preprocessed/answers_study1.csv")
question_columns = df_questions["Question"].unique().tolist()
answers_study1 = answers_study1[question_columns]
fraction_yes = answers_study1.mean().reset_index(name="fraction_yes")
fraction_yes = fraction_yes.rename(
    columns={"index": "Question", "fraction_yes": "Fraction Yes"}
)
fraction_yes = fraction_yes.merge(df_questions, on="Question", how="inner")
fraction_yes = fraction_yes.sort_values("Fraction Yes", ascending=False)

# to validate mode we need how close to either 0 or 1.
# and then we need the average of this across all questions.
import numpy as np

fraction_yes["Fraction No"] = 1 - fraction_yes["Fraction Yes"]
fraction_yes["Fraction distance"] = np.max(
    fraction_yes[["Fraction Yes", "Fraction No"]], axis=1
)
fraction_yes["Fraction distance"].mean()  # roughly matches what we observe.

""" 
some really hard because almost always yes: 
- supernatural beings present: 99.71 yes 
- cultural contact: 99.43% yes
- belief in afterlife: 97.41% yes

some really hard because almost always no:
- sacrifice children: 1.44% yes
- adult sacrifice: 2.89% yes 
"""

# try some of the first plots without the extreme questions #
remove_questions = ["X4827", "X4654", "X4780", "X4776", "X5137", "X5132"]
df_long_balanced = df_long[~df_long["Question"].isin(remove_questions)]
df_long_balanced = df_long_balanced.sort_values("Method")

# overall metrics plot #
multiple_lineplots(
    df=df_long_balanced,
    metric="values",
    hue="Method",
    grid="metric",
    ncol_legend=4,
    outpath=outpath,
    outname="overall_metrics_subset.png",
)
