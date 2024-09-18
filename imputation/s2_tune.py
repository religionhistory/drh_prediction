import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import json
from sklearn.linear_model import BayesianRidge

# Import XGBoost
from xgboost import XGBRegressor
from imputation_functions import impute_data, evaluate_imputation
from hyper_parameters import (
    param_grid_rf,
    param_grid_xgb,
    param_grid_knn,
    param_grid_bayesian_ridge,
)

# Load the original data and the tuning dataset
data_original = pd.read_csv("../data/preprocessed/answers_study2.csv")
data_missing = pd.read_csv("../data/study2/hyperparams/NA_MCAR_15_1.csv")
df_relationships = pd.read_csv("../data/preprocessed/question_level_study2.csv")

# Define non-question predictors
imputation_columns = data_original.filter(regex="^Q_").columns
non_question_predictors = data_original.columns.difference(imputation_columns)

# Define hyperparameter grids and imputation methods as before
imputation_methods = {
    "IterativeImputer_RandomForest": {
        "imputer_class": IterativeImputer,
        "estimator_class": RandomForestClassifier,
        "param_grid": param_grid_rf,
    },
    "IterativeImputer_XGBoost": {
        "imputer_class": IterativeImputer,
        "estimator_class": XGBRegressor,
        "param_grid": param_grid_xgb,
    },
    "KNNImputer": {
        "imputer_class": KNNImputer,
        "estimator_class": None,
        "param_grid": param_grid_knn,
    },
    "IterativeImputer_BayesianRidge": {
        "imputer_class": IterativeImputer,
        "estimator_class": BayesianRidge,
        "param_grid": param_grid_bayesian_ridge,
    },
}

# actually see whether we can get this to work ... #

# Perform tuning for each imputation method
for imputer_name, imputer_info in imputation_methods.items():
    imputer_class = imputer_info["imputer_class"]
    param_grid = imputer_info["param_grid"]
    estimator_class = imputer_info.get("estimator_class", None)

    results_df, best_params = tune_hierarchical_imputer(
        imputer_name=imputer_name,
        imputer_class=imputer_class,
        estimator_class=estimator_class,
        param_grid=param_grid,
        data_missing=data_missing,
        data_original=data_original,
        df_relationships=df_relationships,
        non_question_predictors=non_question_predictors,
        imputation_columns=imputation_columns,
    )

    # Save the results to CSV
    results_df.to_csv(
        f"../data/study2/hyperparams/imputation_{imputer_name}_tuning_results.csv",
        index=False,
    )

    # Save best parameters to JSON
    with open(f"../data/study2/hyperparams/best_{imputer_name}_params.json", "w") as f:
        json.dump(best_params, f)
