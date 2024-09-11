"""
2024-09-11 VMP

Create datasets for two studies.
- Study 1: 15 most answered questions
- Study 2: 100 most answered questions
"""

import numpy as np
import pandas as pd

# load answers
questions = pd.read_csv("../data/preprocessed/answers.csv")


# function to get the n most answered questions
def get_count_answers(
    questions: pd.DataFrame, n: int, min_mixing: float = 0.05
) -> pd.DataFrame:
    """
    For min_mixing = 0 get n most answered questions.
    For min_mixing > 0 get n most answered with mixing (yes-no) above min_mixing.
    """
    answer_mean = (
        questions.groupby("question_id")["answer_value"]
        .mean()
        .reset_index(name="answer_mean")
    )
    answer_mean = answer_mean[
        (answer_mean["answer_mean"] > min_mixing)
        & (answer_mean["answer_mean"] < 1 - min_mixing)
    ]
    answer_mean = answer_mean["question_id"].unique().tolist()
    questions = questions[questions["question_id"].isin(answer_mean)]
    count_answers = questions.groupby("question_id").size().reset_index(name="count")
    count_answers = count_answers.sort_values(by="count", ascending=False)
    count_answers = count_answers.head(n)
    return count_answers


# get the 15 and 100 most answered questions
small_dataset = get_count_answers(questions, 15)
large_dataset = get_count_answers(questions, 100)

questions = questions[["entry_id", "question_id", "answer_value"]].drop_duplicates()
questions_small = questions[questions["question_id"].isin(small_dataset["question_id"])]
questions_large = questions[questions["question_id"].isin(large_dataset["question_id"])]


# function to get the entries with more than threshold answers
def get_count_entries(questions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Get the entries with more than threshold answers.
    """
    n_questions = questions["question_id"].nunique()
    count_entries = questions.groupby("entry_id").size().reset_index(name="count")
    count_entries["percentage"] = count_entries["count"] / n_questions
    count_entries = count_entries[count_entries["percentage"] >= threshold]
    unique_entries = count_entries["entry_id"].unique().tolist()
    questions = questions[questions["entry_id"].isin(unique_entries)]
    return questions


# require 100% answers for the small dataset
questions_small = get_count_entries(questions_small, threshold=1.0)

# require 50% answers for the large dataset
questions_large = get_count_entries(questions_large, threshold=0.5)

### add question level ###
question_level = pd.read_csv("../data/preprocessed/question_level.csv")


def process_question_level(question_level, question_subset):
    question_subset = question_subset[["question_id"]].drop_duplicates()
    question_subset = question_subset.merge(
        question_level, on="question_id", how="inner"
    )
    return question_subset


question_level_small = process_question_level(question_level, questions_small)
question_level_large = process_question_level(question_level, questions_large)

question_level_small.to_csv(
    "../data/preprocessed/question_level_study1.csv", index=False
)
question_level_large.to_csv(
    "../data/preprocessed/question_level_study2.csv", index=False
)


### pivot the questions ###
# load entries
entry_metadata = pd.read_csv("../data/preprocessed/entry_metadata.csv")


def pivot_questions(questions, entry_metadata):
    questions_wide = questions.pivot_table(
        index="entry_id",
        columns="question_id",
        values="answer_value",
        fill_value=np.nan,
    )
    questions_wide = questions_wide.reset_index()
    questions_wide = entry_metadata.merge(questions_wide, on="entry_id", how="inner")
    return questions_wide


pivot_small = pivot_questions(questions_small, entry_metadata)
pivot_large = pivot_questions(questions_large, entry_metadata)


# prefix question columns with Q
def prefix_question_columns(df):
    modified_names = [f"Q_{col}" if isinstance(col, int) else col for col in df.columns]
    df.columns = modified_names
    return df


pivot_small = prefix_question_columns(pivot_small)
pivot_large = prefix_question_columns(pivot_large)

# save data
pivot_small.to_csv("../data/preprocessed/answers_study1.csv", index=False)
pivot_large.to_csv("../data/preprocessed/answers_study2.csv", index=False)

# save id vars (same for pivot_small and pivot_large)
# used in add_NA_0.R
id_columns = [col for col in pivot_small.columns if not col.startswith("Q_")]
file_path = "../imputation/id_vars.txt"
with open(file_path, "w") as file:
    for col in id_columns:
        file.write(col + "\n")
