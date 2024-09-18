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


def impute_and_evaluate(imputer, data_missing, data_original, imputation_columns):
    """
    Performs imputation on the given data and evaluates the imputed data against the original data using F1 score.
    """
    # Perform imputation
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)

    # Evaluate the performance
    data_missing_questions = data_missing[imputation_columns]
    data_original_questions = data_original[imputation_columns]
    data_imputed_questions = data_imputed[imputation_columns]

    # some methods (e.g. XGB) return non-binary values
    data_imputed_questions = data_imputed_questions.map(lambda x: 1 if x > 0.5 else 0)

    # Identify where missing values were introduced
    mask = data_missing_questions.isnull()
    true_values = data_original_questions[mask]
    imputed_values = data_imputed_questions[mask]

    f1_scores = []
    f1_scores_dict = {}

    for column in imputation_columns:
        y_true = true_values[column].dropna()
        y_pred = imputed_values[column].loc[y_true.index]
        if y_true.empty or y_true.nunique() == 1 or y_pred.nunique() == 1:
            continue  # Skip if no data to compare or only one class present
        score = f1_score(y_true, y_pred.round())
        f1_scores.append(score)
        f1_scores_dict[column] = score

    if f1_scores:
        avg_f1 = np.mean(f1_scores)
    else:
        avg_f1 = None  # Or set to 0 or np.nan as appropriate

    return avg_f1, f1_scores_dict


def tune_imputer(
    imputer_name,
    imputer_class,
    estimator_class,
    param_grid,
    data_missing,
    data_original,
    imputation_columns,
):
    """
    Performs hyperparameter tuning for a given imputer and evaluates performance.
    """
    results = []
    best_score = -np.inf
    best_params = None

    # Generate the parameter grid
    grid = list(ParameterGrid(param_grid))

    for params in grid:
        print(f"Testing parameters for {imputer_name}: {params}")
        # Initialize the imputer
        if imputer_name == "IterativeImputer_BayesianRidge":
            # Set up the estimator with params
            estimator = estimator_class(**params)
            imputer = imputer_class(
                estimator=estimator,
                max_iter=10,
                random_state=0,
                tol=1e-3,
            )
        elif imputer_name in [
            "IterativeImputer_RandomForest",
            "IterativeImputer_XGBoost",
        ]:
            # Set up the estimator with params
            estimator = estimator_class(**params, random_state=0, n_jobs=-1)
            imputer = imputer_class(
                estimator=estimator,
                max_iter=10,
                random_state=0,
                tol=1e-3,
            )
        elif imputer_name == "KNNImputer":
            imputer = imputer_class(**params)
        else:
            continue  # Skip if imputer_name not recognized

        # Call the impute_and_evaluate function
        avg_f1, f1_scores_dict = impute_and_evaluate(
            imputer=imputer,
            data_missing=data_missing,
            data_original=data_original,
            imputation_columns=imputation_columns,
        )

        if avg_f1 is not None:
            print(f"Average F1 score: {avg_f1}")

            # Store results for this parameter set
            result = params.copy()
            result["imputer_name"] = imputer_name
            result["avg_f1"] = avg_f1
            # consider whether we need column below.
            # result["f1_scores"] = f1_scores_dict
            results.append(result)

            # Update best parameters if current score is better
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = params.copy()
                print(
                    f"New best parameters for {imputer_name}: {best_params} with F1 score: {best_score}"
                )
        else:
            print("No valid F1 scores calculated for this parameter set.")

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df, best_params


# Hyperparameter grid for RandomForestClassifier used in IterativeImputer
param_grid_rf = {
    "n_estimators": [20, 50, 100, 200],  # default: 100
    "max_depth": [None, 3, 5, 7],  # default: None
    "max_features": ["sqrt", 0.3],  # default: "sqrt"
    "min_samples_split": [2, 5],  # default
    "min_samples_leaf": [1, 2],  # default
    "bootstrap": [True],  # default
}

# Hyperparameter grid for KNNImputer
param_grid_knn = {
    "n_neighbors": [2, 3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["nan_euclidean"],
}

# Hyperparameter grid for XGBRegressor used in IterativeImputer
param_grid_xgb = {
    "n_estimators": [20, 50, 100, 200],
    "max_depth": [None, 3, 5, 7],  # complexity
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [1.0],  # no subsampling
    "colsample_bytree": [1.0],  # no subsampling
    "objective": ["reg:squarederror"],
}

# bayesian ridge as well
param_grid_bayesian_ridge = {
    "alpha_1": [1e-6, 1e-5],
    "lambda_1": [1e-6, 1e-5],
    "fit_intercept": [True, False],
}

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
original_data = pd.read_csv("../data/preprocessed/answers_study1.csv")
imputation_columns = original_data.filter(regex="Q_").columns

# Load the tuning dataset
tuning_file = "NA_MCAR_25_1.csv"  # Replace with your chosen file
data_missing = pd.read_csv(f"../data/study1/hyperparams/{tuning_file}")

all_results = []
best_params_all = {}

for imputer_name, imputer_info in imputation_methods.items():
    imputer_class = imputer_info["imputer_class"]
    param_grid = imputer_info["param_grid"]
    estimator_class = imputer_info.get("estimator_class", None)

    # Perform hyperparameter tuning and evaluation
    results_df, best_params = tune_imputer(
        imputer_name=imputer_name,
        imputer_class=imputer_class,
        estimator_class=estimator_class,
        param_grid=param_grid,
        data_missing=data_missing,
        data_original=original_data,
        imputation_columns=imputation_columns,
    )

    # Save the results to CSV
    results_df.to_csv(
        f"../data/study1/hyperparams/imputation_{imputer_name}_tuning_results.csv",
        index=False,
    )

    # Save best parameters to JSON
    with open(f"../data/study1/hyperparams/best_{imputer_name}_params.json", "w") as f:
        json.dump(best_params, f)
