# also shows that in the final analysis we are working with
# n = 55700 answers (not 150.000)
# but there must also be fewer conflicts then.

import pandas as pd
import numpy as np

# load answers and answers imputed
answers_original = pd.read_csv("../data/preprocessed/answers_study2.csv")
answers_imputed = pd.read_csv("../data/study3/imputed_data.csv")

# melt both to long
question_vars = [x for x in answers_original.columns if x.startswith("Q_")]
answers_orig_long = pd.melt(
    answers_original,
    id_vars=["entry_id"],
    value_vars=question_vars,
    var_name="question_id",
    value_name="answer",
)
answers_imputed_long = pd.melt(
    answers_imputed,
    id_vars=["entry_id"],
    value_vars=question_vars,
    var_name="question_id",
    value_name="answer",
)

# now merge these
answers_orig_long = answers_orig_long.rename(columns={"answer": "answer_original"})
answers_imputed_long = answers_imputed_long.rename(
    columns={"answer": "answer_predicted"}
)
answers_long = answers_orig_long.merge(
    answers_imputed_long, on=["entry_id", "question_id"], how="inner"
)

# now we need to add entry name
entry_data = pd.read_csv("../data/raw/entry_data.csv")
entry_data = entry_data[["entry_id", "entry_name"]].drop_duplicates()
answers_long = answers_long.merge(entry_data, on="entry_id", how="inner")

# now we need to add question text
from constants import short_labels

answers_long["question_name"] = answers_long["question_id"].apply(
    lambda x: short_labels[x]
)
answers_long = answers_long.sort_values("entry_id")
answers_long = answers_long[
    [
        "entry_id",
        "entry_name",
        "question_id",
        "question_name",
        "answer_original",
        "answer_predicted",
    ]
]

# save this both with and without the original values
answers_long.to_csv("../tables/s3_answers_filled.csv", index=False)
answers_long_small = answers_long[answers_long["answer_original"].isna()]
answers_long_small = answers_long_small.drop(columns=["answer_original"])
answers_long_small.to_csv("../tables/s3_answers_predicted.csv", index=False)

# try proselytizing
proselytizing = answers_long[answers_long["question_name"] == "Proselytizing"]
proselytizing_missing = proselytizing[proselytizing["answer_original"].isna()]
proselytizing_missing = proselytizing_missing[
    ["entry_id", "entry_name", "answer_predicted"]
]
proselytizing_missing["answer_predicted"] = np.where(
    proselytizing_missing["answer_predicted"] > 0.5, "Yes", "No"
)
proselytizing_missing.to_latex("../tables/s3_proselytizing_missing.tex", index=False)
