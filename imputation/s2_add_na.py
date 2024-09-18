import numpy as np
import pandas as pd
import torch
import wget
import os

# download the utils.py file from the github repository
url = "https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py"
filename = "utils.py"

if not os.path.exists(filename):
    wget.download(url)

from utils import (
    MAR_mask,
    MNAR_mask_logistic,
    MNAR_mask_quantiles,
    MNAR_self_mask_logistic,
)


# function taken from: https://rmisstastic.netlify.app/how-to/python/generate_html/how%20to%20generate%20missing%20values
def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str,
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str,
        For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.

    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)

    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1 - p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()

    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan

    return {"X_init": X.double(), "X_incomp": X_nas.double(), "mask": mask}


# for sub-questions
def conditional_update(df, column_to_update, reference_column, condition, update_value):
    """
    Updates values in one column based on a condition applied to another column.

    Parameters:
        df (pd.DataFrame): The dataframe to operate on.
        column_to_update (str): The name of the column to update.
        reference_column (str): The name of the column to apply the condition to.
        condition (callable): A function that takes a Series and returns a boolean Series.
        update_value (any or callable): The value to set or a function that returns the value to set based on the reference column.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Create a mask where the condition is True
    mask = condition(df[reference_column])

    # Apply update: if update_value is callable, use it to generate values, else use it directly
    if callable(update_value):
        df.loc[mask, column_to_update] = update_value(df.loc[mask, reference_column])
    else:
        df.loc[mask, column_to_update] = update_value

    return df


# for sub-questions
def update_sub_questions(data, question_levels, condition, update_value):
    # find pairs of questions
    data_copy = data.copy()
    sub_questions = question_levels[question_levels["question_level"] > 1]
    sub_questions = sub_questions.sort_values("question_level")
    question_pairs = sub_questions[["question_id", "parent_question_id"]].to_numpy()
    # update each pair in turn
    for pair in question_pairs:
        subquestion = "Q_{}".format(pair[0])
        superquestion = "Q_{}".format(pair[1])
        # the super-question has to also be in the data
        if superquestion in data_copy.columns:
            data_copy = conditional_update(
                data_copy, subquestion, superquestion, condition, update_value
            )
    return data_copy


def gather_missingness(question_levels, complete_answers, imputations):
    X_missing = imputations["X_incomp"]
    X_missing = pd.DataFrame(
        X_missing.numpy(), columns=complete_answers.filter(regex="^Q_").columns
    )
    # update sub-questions
    X_missing = update_sub_questions(
        X_missing, question_levels, lambda x: x.isna(), np.nan
    )
    # bind with id vars & save
    X_missing = pd.concat([id_var, X_missing], axis=1)
    return X_missing


def check_relationships(df_data, df_relationships):
    """
    Optimized function to check hierarchical relationships in df_data.
    """
    violations = []

    # Filter relationships where both questions exist in df_data
    valid_relationships = df_relationships[
        df_relationships["question_id"].isin(df_data.columns)
        & df_relationships["parent_question_id"].isin(df_data.columns)
    ]

    for _, row in valid_relationships.iterrows():
        sub_question = row["question_id"]
        super_question = row["parent_question_id"]

        # Create a mask for violation
        mask_violation = df_data[super_question].isna() & df_data[sub_question].notna()
        violating_indices = df_data.index[mask_violation].tolist()

        if violating_indices:
            violation_df = pd.DataFrame(
                {
                    "index": violating_indices,
                    "super_question": super_question,
                    "super_question_value": df_data.loc[
                        violating_indices, super_question
                    ],
                    "sub_question": sub_question,
                    "sub_question_value": df_data.loc[violating_indices, sub_question],
                }
            )
            violations.append(violation_df)

    if violations:
        violations_df = pd.concat(violations, ignore_index=True)
    else:
        violations_df = pd.DataFrame(
            columns=[
                "index",
                "super_question",
                "super_question_value",
                "sub_question",
                "sub_question_value",
            ]
        )
    return violations_df


# load answers
answers = pd.read_csv("../data/preprocessed/answers_study2.csv")

# split into questions and non-questions
question_var = answers.filter(regex="^Q_")
id_var = answers[answers.columns.difference(question_var.columns)]

# now make integer and ready for imputation
question_var = question_var.to_numpy()

# also need question levels
question_level = pd.read_csv("../data/preprocessed/question_level_study2.csv")


# grid of missingness
def MCAR_missingness(
    complete_answers,
    question_var,
    question_levels,
    iter,
    p_miss=[0.1, 0.2],
    folder="additional_NA",
):
    for p in p_miss:
        # generate
        percent = int(p * 100)
        mcar_data = produce_NA(question_var, p, mecha="MCAR")
        # also adding missingness to sub-questions if super-question is missing
        X_missing = gather_missingness(question_levels, complete_answers, mcar_data)
        # validating that sub-questions are missing when super-question is missing
        res = check_relationships(X_missing, question_level)
        if res.empty:
            X_missing.to_csv(
                f"../data/study2/{folder}/NA_MCAR_{percent}_{iter}.csv", index=False
            )
        else:
            print(f"Violations found for {percent}% missingness in iteration {iter}")


# the data sets for imputation / prediction
for iter in range(2):
    MCAR_missingness(answers, question_var, question_level, iter)

# the data sets for tuning hyperparameters
MCAR_missingness(
    answers,
    question_var,
    question_level,
    1,
    p_miss=[0.15],
    folder="hyperparams",
)
