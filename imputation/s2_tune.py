import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
import json
from sklearn.linear_model import BayesianRidge

# Import XGBoost
from xgboost import XGBRegressor
from imputation_functions import (
    create_imputer,
    impute_data,
    evaluate_imputation,
    hierarchical_imputation,
    data_integrity_check,
    hierarchical_relationship_check,
)
from hyper_parameters import (
    param_grid_rf,
    param_grid_xgb,
    param_grid_knn,
    param_grid_bayesian_ridge,
)

study = "study2"

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

# Load the original data and the tuning dataset
data_original = pd.read_csv(f"../data/preprocessed/answers_{study}.csv")
data_missing = pd.read_csv(f"../data/{study}/hyperparams/NA_MCAR_15_1.csv")
df_relationships = pd.read_csv(f"../data/preprocessed/question_level_{study}.csv")

# adjust the question IDs
df_relationships["question_id"] = "Q_" + df_relationships["question_id"].astype(str)
df_relationships["parent_question_id"] = "Q_" + df_relationships[
    "parent_question_id"
].astype(str)

# Define non-question predictors
imputation_columns = data_original.filter(regex="^Q_").columns
non_question_predictors = data_original.columns.difference(imputation_columns)

best_params_all = {}
all_results = []

for imputer_name, imputer_info in imputation_methods.items():
    imputer_class = imputer_info["imputer_class"]
    estimator_class = imputer_info.get("estimator_class", None)
    param_grid = imputer_info["param_grid"]

    # Initialize variables to keep track of best parameters
    best_score = -np.inf
    best_params = None
    results = []

    # Generate the parameter grid
    grid = list(ParameterGrid(param_grid))

    for params in grid:
        print(f"Testing parameters for {imputer_name}: {params}")

        # Initialize the imputer with current parameters
        imputer = create_imputer(imputer_name, imputer_class, estimator_class, params)

        # Perform hierarchical imputation
        data_imputed = hierarchical_imputation(
            df_relationships=df_relationships,
            non_question_predictors=non_question_predictors,
            data_missing=data_missing,
            imputer=imputer,
        )

        data_integrity_check(data_missing, data_imputed)
        hierarchical_relationship_check(data_imputed, df_relationships)

        # After all levels are processed, evaluate the imputed data
        avg_metrics, metrics_dict = evaluate_imputation(
            data_imputed=data_imputed,
            data_original=data_original,
            data_missing=data_missing,
            imputation_columns=imputation_columns,
        )

        # Use F1 score as the metric to optimize
        current_score = avg_metrics["f1_score"]

        # Store results
        result = params.copy()
        result["imputer_name"] = imputer_name
        result["avg_f1_score"] = current_score
        results.append(result)

        if current_score > best_score:
            best_score = current_score
            best_params = params.copy()
            print(
                f"New best parameters for {imputer_name}: {best_params} with score: {best_score}"
            )

    # Save the results for this imputer
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        f"../data/{study}/hyperparams/imputation_{imputer_name}.csv", index=False
    )

    # Optionally, save best parameters to a JSON file
    with open(f"../data/{study}/hyperparams/best_params_{imputer_name}.json", "w") as f:
        json.dump(best_params, f)
