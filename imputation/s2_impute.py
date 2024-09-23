import pandas as pd
import numpy as np
import json
import glob
import os
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor

# Import custom functions
from imputation_functions import (
    impute_data,
    evaluate_imputation,
    hierarchical_imputation,
    data_integrity_check,
    hierarchical_relationship_check,
)

study = "study2"

# Load the original data
data_original = pd.read_csv(f"../data/preprocessed/answers_{study}.csv")

# Load relationships and adjust question IDs
df_relationships = pd.read_csv(f"../data/preprocessed/question_level_{study}.csv")
df_relationships["question_id"] = "Q_" + df_relationships["question_id"].astype(str)
df_relationships["parent_question_id"] = "Q_" + df_relationships[
    "parent_question_id"
].astype(str)

# Define imputation columns and non-question predictors
imputation_columns = data_original.filter(regex="^Q_").columns.tolist()
non_question_predictors = data_original.columns.difference(imputation_columns).tolist()

# Load best parameters for each imputer
best_params = {}

# Define the list of imputers you have
imputer_names = [
    "KNNImputer",
    "IterativeImputer_RandomForest",
    "IterativeImputer_XGBoost",
    "IterativeImputer_BayesianRidge",
]

# Load best parameters for each imputer
for imputer_name in imputer_names:
    try:
        with open(
            f"../data/{study}/hyperparams/best_params_{imputer_name}.json", "r"
        ) as f:
            best_params[imputer_name] = json.load(f)
    except FileNotFoundError:
        print(f"Best parameters file for {imputer_name} not found.")
        continue

# Create imputers dictionary
imputers = {
    "mode": SimpleImputer(strategy="most_frequent"),
    "knn": KNNImputer(**best_params.get("KNNImputer", {})),
    "ridge": IterativeImputer(
        estimator=BayesianRidge(
            **best_params.get("IterativeImputer_BayesianRidge", {})
        ),
        max_iter=10,
        random_state=0,
        tol=1e-3,
    ),
    "rf": IterativeImputer(
        estimator=RandomForestClassifier(
            **best_params.get("IterativeImputer_RandomForest", {}), random_state=0
        ),
        max_iter=10,
        random_state=0,
        tol=1e-3,
    ),
    "xgb": IterativeImputer(
        estimator=XGBRegressor(
            **best_params.get("IterativeImputer_XGBoost", {}), random_state=0
        ),
        max_iter=10,
        random_state=0,
        tol=1e-3,
    ),
}

# Get list of datasets with missing values
missing_data_files = glob.glob(f"../data/{study}/additional_NA/*.csv")

# Regular expression to extract missing type and percentage from file names
import re

file_regex = r"NA_(MCAR|MAR|MNAR)_(\d+)_(\d+).csv"


# Function to run imputation and evaluation for a given imputer
def run_imputation(
    missing_data_files,
    imputer_name,
    imputer,
    data_original,
    df_relationships,
    non_question_predictors,
    imputation_columns,
):
    # Output folder for imputed datasets
    output_folder = f"../data/{study}/{imputer_name}"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a list to store evaluation metrics across all files
    evaluation_rows = []

    for data_file in missing_data_files:
        print(f"Imputing file {data_file} with {imputer_name}")
        # Extract dataset name for identification
        dataset_name = os.path.basename(data_file)

        # Extract missing type, missing percent, and iteration from file name
        match = re.match(file_regex, dataset_name)
        if match:
            missing_type, missing_percent, iteration = match.groups()
        else:
            print(f"Filename {dataset_name} does not match expected pattern.")
            missing_type, missing_percent, iteration = None, None, None

        # Load the dataset with missing values
        data_missing = pd.read_csv(data_file)

        # Perform hierarchical imputation
        data_imputed = hierarchical_imputation(
            df_relationships=df_relationships,
            non_question_predictors=non_question_predictors,
            data_missing=data_missing,
            imputer=imputer,
        )

        # Verify the imputed data
        data_integrity_check(data_missing, data_imputed)
        hierarchical_relationship_check(data_imputed, df_relationships)

        # Evaluate the imputed data
        avg_metrics, metrics_dict = evaluate_imputation(
            data_imputed=data_imputed,
            data_original=data_original,
            data_missing=data_missing,
            imputation_columns=imputation_columns,
        )

        # Store the per-column metrics
        for column, metrics in metrics_dict.items():
            result = {
                "file": dataset_name,
                "method": imputer_name,
                "missing_type": missing_type,
                "missing_percent": missing_percent,
                "iteration": iteration,
                "column": column,
                "f1_score": metrics.get("f1_score"),
                "accuracy": metrics.get("accuracy"),
                "mcc": metrics.get("mcc"),
            }
            evaluation_rows.append(result)

        # Save the imputed dataset
        imputed_data_filename = os.path.join(output_folder, dataset_name)
        data_imputed.to_csv(imputed_data_filename, index=False)

        print(
            f"Imputation and evaluation for {imputer_name} on {dataset_name} completed."
        )

    # Convert the evaluation results to a DataFrame
    evaluation_results = pd.DataFrame(evaluation_rows)

    # Save the evaluation results for this imputer
    eval_output_folder = f"../data/{study}/evaluation"
    os.makedirs(eval_output_folder, exist_ok=True)
    evaluation_results.to_csv(
        os.path.join(eval_output_folder, f"metrics_{imputer_name}.csv"), index=False
    )

    print(f"Evaluation results saved for imputer: {imputer_name}")


# Run imputation and evaluation for each imputer
for imputer_name, imputer in imputers.items():
    run_imputation(
        missing_data_files=missing_data_files,
        imputer_name=imputer_name,
        imputer=imputer,
        data_original=data_original,
        df_relationships=df_relationships,
        non_question_predictors=non_question_predictors,
        imputation_columns=imputation_columns,
    )
