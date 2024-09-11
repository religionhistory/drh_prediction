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


# load answers
answers_study1 = pd.read_csv("../data/preprocessed/answers_study1.csv")

# split into questions and non-questions
question_var = answers_study1.filter(regex="^Q_")
id_var = answers_study1[answers_study1.columns.difference(question_var.columns)]

# now make integer and ready for imputation
question_var = question_var.astype(int)
question_var = question_var.to_numpy()

# also need question levels
question_level_study1 = pd.read_csv("../data/preprocessed/question_level_study1.csv")

# grid of missingness
p_miss = [0.1, 0.2, 0.3, 0.4, 0.5]

for p in p_miss:
    p_ = p * 1.25  # 1 / (1 - 0.2)
    # generate missing data
    mcar_data = produce_NA(question_var, p, mecha="MCAR")
    mar_data = produce_NA(question_var, p_, mecha="MAR", p_obs=0.2)
    mnar_data = produce_NA(question_var, p, mecha="MNAR", opt="logistic", p_obs=0.5)

    for data, mecha in zip([mcar_data, mar_data, mnar_data], ["MCAR", "MAR", "MNAR"]):
        X_missing = data["X_incomp"]
        X_missing = pd.DataFrame(
            X_missing.numpy(), columns=answers_study1.filter(regex="^Q_").columns
        )
        # check missingness
        print(f"p_miss: {p}")
        print(f"miss: {np.round(X_missing.isna().mean().mean(), 2)}")
        # update sub-questions
        X_missing = update_sub_questions(
            X_missing, question_level_study1, lambda x: x.isna(), np.nan
        )
        # bind with id vars & save
        X_missing = pd.concat([id_var, X_missing], axis=1)
        X_missing.to_csv(f"../data/study1/additional_NA/{mecha}_{p}.csv", index=False)
