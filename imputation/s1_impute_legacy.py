""" 
to-do: 
make a general pipeline that makes sense.
do that for the random forest classifier.
then try to figure out what actually works.
would be cool to try XGBoost as well. 
not sure about hyper parameter tuning. 

need to fix the levels that we are imputing at. 
just as we did in R. 

In this case we do not need to worry about imputation at different levels.
We do have one sub-question but the super-question is not in the sample. 
(This is because it has too little variation; less than 5%).
We will have to solve this for study 2. 

Think next step really should be figuring out how to do: 
1) hyper-parameter tuning.
2) evaluating on large data (also adding NAN probably). 
3) implementing the child thing for the large data. 
4) how to report the SHAP values.
5) writing it up. 
"""

# https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html
# https://scikit-learn.org/stable/auto_examples/impute/plot_iterative_imputer_variants_comparison.html

from sklearn.experimental import enable_iterative_imputer
import pandas as pd
import os
import json
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
import re

# Load full data
data_full = pd.read_csv("../data/preprocessed/answers_study1.csv")
question_columns = data_full.filter(regex="Q_").columns
original_data = data_full.drop(columns=question_columns)
file_regex = r"NA_(MCAR|MAR|MNAR)_(\d+)_(\d+).csv"

# List of files with missingness
files_missingness = os.listdir("../data/study1/additional_NA")

# load best params from hyperparameter tuning
best_params = {}

# Load best parameters for KNNImputer
with open("../data/study1/hyperparams/best_KNNImputer_params.json", "r") as f:
    best_params["KNNImputer"] = json.load(f)

# Load best parameters for RandomForest
with open(
    "../data/study1/hyperparams/best_IterativeImputer_RandomForest_params.json", "r"
) as f:
    best_params["IterativeImputer_RandomForest"] = json.load(f)

# Load best parameters for XGBoost
with open(
    "../data/study1/hyperparams/best_IterativeImputer_XGBoost_params.json", "r"
) as f:
    best_params["IterativeImputer_XGBoost"] = json.load(f)

# Load best parameters for BayesianRidge
with open(
    "../data/study1/hyperparams/best_IterativeImputer_BayesianRidge_params.json", "r"
) as f:
    best_params["IterativeImputer_BayesianRidge"] = json.load(f)

imputers = {
    "mode": SimpleImputer(strategy="most_frequent"),
    "knn": KNNImputer(**best_params["KNNImputer"]),
    "ridge": IterativeImputer(
        estimator=BayesianRidge(**best_params["IterativeImputer_BayesianRidge"]),
        max_iter=10,
        random_state=0,
    ),
    "rf": IterativeImputer(
        estimator=RandomForestClassifier(
            **best_params["IterativeImputer_RandomForest"], random_state=0
        ),
        max_iter=10,
        random_state=0,
    ),
    "xgb": IterativeImputer(
        estimator=XGBRegressor(
            **best_params["IterativeImputer_XGBoost"], random_state=0
        ),
        max_iter=10,
        random_state=0,
    ),
}


def run_imputation(
    files_missingness, imputer_name, imputer, data_full, original_data, question_columns
):
    output_folder = f"../data/study1/{imputer_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a DataFrame to store evaluation metrics across all files
    evaluation_rows = []

    for file_missing in files_missingness:
        print(f"Imputing file {file_missing} with {imputer_name}")
        missing_type, missing_percent, iter = re.match(
            file_regex, file_missing
        ).groups()
        data_missing = pd.read_csv(f"../data/study1/additional_NA/{file_missing}")
        data_imputed = imputer.fit_transform(data_missing)
        data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)

        # Round imputed values for question columns to integers.
        # Cannot just be round() because VERY rare cases go to -1 or 2.
        data_imputed[question_columns] = data_imputed[question_columns].map(
            lambda x: 1 if x > 0.5 else 0
        )

        # Replace original data (non-question columns)
        data_imputed[original_data.columns] = original_data[original_data.columns]

        # Save the imputed dataset
        output_path = os.path.join(output_folder, file_missing)
        data_imputed.to_csv(output_path, index=False)

        # Compute evaluation metrics
        # Only consider positions where data_missing is NaN (i.e., where data was missing)
        mask = data_missing.isnull()
        # We'll compare data_imputed and original_data at these positions
        imputed_values = data_imputed[mask]
        true_values = data_full[mask]

        for column in question_columns:
            # If there were missing values in this column
            if mask[column].any():
                y_true = true_values[column].dropna()
                y_pred = imputed_values[column].loc[y_true.index]
                if y_true.nunique() < 2 or y_pred.nunique() < 2:
                    # Cannot compute some metrics with less than two classes
                    acc = accuracy_score(y_true, y_pred)
                    f1 = np.nan
                    mcc = np.nan
                else:
                    f1 = f1_score(y_true, y_pred)
                    acc = accuracy_score(y_true, y_pred)
                    mcc = matthews_corrcoef(y_true, y_pred)
            else:
                # No missing values in this column
                f1 = np.nan
                acc = np.nan
                mcc = np.nan

            # Store metrics
            evaluation_rows.append(
                {
                    "file": file_missing,
                    "method": imputer_name,
                    "missing_type": missing_type,
                    "missing_percent": missing_percent,
                    "iteration": iter,
                    "column": column,
                    "imputer": imputer_name,
                    "f1_score": f1,
                    "accuracy": acc,
                    "mcc": mcc,
                }
            )

    # convert list to dataframe
    evaluation_results = pd.DataFrame(evaluation_rows)
    # Save evaluation results for this imputer
    eval_output_folder = f"../data/study1/evaluation"
    os.makedirs(eval_output_folder, exist_ok=True)
    evaluation_results.to_csv(
        os.path.join(eval_output_folder, f"metrics_{imputer_name}.csv"), index=False
    )


# Run imputation and evaluation for each imputer
for imputer_name, imputer in imputers.items():
    run_imputation(
        files_missingness=files_missingness,
        imputer_name=imputer_name,
        imputer=imputer,
        original_data=original_data,
        data_full=data_full,
        question_columns=question_columns,
    )
