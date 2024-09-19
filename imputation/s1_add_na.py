"""
VMP 2023-10-18
Do not need the mechanism for sub-questions that we do in study 2. 
This is because the only sub-question here is without the super-question
because the super-question has less than 10% variation.
"""

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


def gather_missingness(complete_answers, imputations):
    X_missing = imputations["X_incomp"]
    X_missing = pd.DataFrame(
        X_missing.numpy(), columns=complete_answers.filter(regex="^Q_").columns
    )
    # bind with id vars & save
    X_missing = pd.concat([id_var, X_missing], axis=1)
    return X_missing


study = "study1"

# load answers
answers = pd.read_csv(f"../data/preprocessed/answers_{study}.csv")

# split into questions and non-questions
question_var = answers.filter(regex="^Q_")
id_var = answers[answers.columns.difference(question_var.columns)]

# now make integer and ready for imputation
question_var = question_var.astype(int)
question_var = question_var.to_numpy()


# grid of missingness
def MCAR_missingness(
    complete_answers,
    question_var,
    question_levels,
    iter,
    p_miss=[0.1, 0.2, 0.3, 0.4, 0.5],
    folder="additional_NA",
):
    for p in p_miss:
        # generate
        percent = int(p * 100)
        mcar_data = produce_NA(question_var, p, mecha="MCAR")
        X_missing = gather_missingness(question_levels, complete_answers, mcar_data)
        X_missing.to_csv(
            f"../data/{study}/{folder}/NA_MCAR_{percent}_{iter}.csv", index=False
        )


def MAR_missingness(
    complete_answers,
    question_var,
    question_levels,
    iter,
    p_miss=[0.1, 0.2, 0.3, 0.4, 0.5],
    folder="additional_NA",
):
    for p in p_miss:
        p_ = p * 1.25  # 1 / (1 - 0.2)
        percent = int(p * 100)  # for saving
        # generate missing data
        mar_data = produce_NA(question_var, p_, mecha="MAR", p_obs=0.2)
        X_missing = gather_missingness(question_levels, complete_answers, mar_data)
        X_missing.to_csv(
            f"../data/{study}/{folder}/NA_MAR_{percent}_{iter}.csv", index=False
        )


def MNAR_missingness(
    complete_answers,
    question_var,
    question_levels,
    iter,
    study,
    p_miss=[0.1, 0.2, 0.3, 0.4, 0.5],
    folder="additional_NA",
):
    for p in p_miss:
        percent = int(p * 100)  # for saving
        # generate missing data
        mnar_data = produce_NA(question_var, p, mecha="MNAR", opt="logistic", p_obs=0.5)
        X_missing = gather_missingness(question_levels, complete_answers, mnar_data)
        X_missing.to_csv(
            f"../data/{study}/{folder}/NA_MNAR_{percent}_{iter}.csv", index=False
        )


# the data sets for imputation / prediction
for iter in range(10):
    MCAR_missingness(answers, question_var, iter, study)
    MAR_missingness(answers, question_var, iter, study)
    MNAR_missingness(answers, question_var, iter, study)

# the data sets for tuning hyperparameters
MCAR_missingness(
    answers,
    question_var,
    1,
    study,
    p_miss=[0.25],
    folder="hyperparams",
)
