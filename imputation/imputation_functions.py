import pandas as pd
import numpy as np


def create_imputer(imputer_name, imputer_class, estimator_class, params):
    """
    Creates an imputer instance based on the imputer name, classes, and parameters.
    """
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
        raise ValueError(f"Imputer name '{imputer_name}' not recognized.")

    return imputer


def impute_data(imputer, data_missing, imputation_columns):
    """
    Performs imputation on the given data using the provided imputer.
    """
    # Perform imputation
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)

    # Handle non-binary outputs (e.g., from regressors)
    data_imputed[imputation_columns] = np.where(
        data_imputed[imputation_columns] > 0.5, 1, 0
    )

    return data_imputed


from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef


def evaluate_imputation(data_imputed, data_original, data_missing, imputation_columns):
    """
    Evaluates the imputed data against the original data using F1 score, accuracy, and MCC.

    Parameters:
    - data_imputed: DataFrame after imputation.
    - data_original: Original DataFrame before missingness was introduced.
    - data_missing: DataFrame with artificially introduced missing values.
    - imputation_columns: List of columns (questions) that were imputed.

    Returns:
    - avg_metrics: Dictionary of average metrics.
    - metrics_dict: Dictionary of metrics for each column.
    """
    # Identify positions where missing values were artificially introduced
    mask = (
        data_missing[imputation_columns].isnull()
        & data_original[imputation_columns].notnull()
    )

    # Extract the true values and imputed values at those positions
    true_values = data_original[imputation_columns][mask]
    imputed_values = data_imputed[imputation_columns][mask]

    metrics = {"f1_score": [], "accuracy": [], "mcc": []}
    metrics_dict = {}

    for column in imputation_columns:
        y_true = true_values[column].dropna()
        y_pred = imputed_values[column].loc[y_true.index]

        metrics_dict[column] = {}

        if y_true.empty or y_pred.empty:
            # Skip this column
            metrics_dict[column]["f1_score"] = np.nan
            metrics_dict[column]["accuracy"] = np.nan
            metrics_dict[column]["mcc"] = np.nan
            continue
        elif y_true.nunique() < 2 or y_pred.nunique() < 2:
            # Not enough classes to compute some metrics
            acc = accuracy_score(y_true, y_pred)
            metrics_dict[column]["f1_score"] = np.nan
            metrics_dict[column]["accuracy"] = acc
            metrics_dict[column]["mcc"] = np.nan
        else:
            f1 = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

            metrics_dict[column]["f1_score"] = f1
            metrics_dict[column]["accuracy"] = acc
            metrics_dict[column]["mcc"] = mcc

        # Append metrics to calculate averages later if needed
        metrics["f1_score"].append(metrics_dict[column]["f1_score"])
        metrics["accuracy"].append(metrics_dict[column]["accuracy"])
        metrics["mcc"].append(metrics_dict[column]["mcc"])

    avg_metrics = {}
    for metric, values in metrics.items():
        # Filter out NaN values
        valid_values = [m for m in values if not np.isnan(m)]
        if valid_values:
            avg_metrics[metric] = np.nanmean(valid_values)
        else:
            avg_metrics[metric] = np.nan  # Set to np.nan if no valid metrics

    return avg_metrics, metrics_dict


def hierarchical_imputation(
    df_relationships, non_question_predictors, data_missing, imputer
):
    """
    Performs hierarchical imputation on the data_missing DataFrame using the provided imputer.
    """
    data_imputed = data_missing.copy()
    levels = sorted(df_relationships["question_level"].unique())

    for level in levels:
        print(f"Processing level {level}")
        level_questions = df_relationships[df_relationships["question_level"] == level][
            "question_id"
        ].tolist()

        # Prepare predictors
        if level == 1:
            predictors = list(non_question_predictors)
        else:
            previous_levels_questions = df_relationships[
                df_relationships["question_level"] < level
            ]["question_id"].tolist()
            predictors = list(non_question_predictors) + previous_levels_questions

        # Apply parent-child rule for levels > 1
        if level > 1:
            level_relationships = df_relationships[
                df_relationships["question_level"] == level
            ]
            for _, row in level_relationships.iterrows():
                child = row["question_id"]
                parent = row["parent_question_id"]
                if parent in data_imputed.columns:
                    parent_zero_mask = data_imputed[parent] == 0
                    data_imputed.loc[parent_zero_mask, child] = (
                        0  # Set child to 0 where parent is 0
                    )

        # Identify missing values to impute
        missing_mask = data_imputed[level_questions].isna()
        if missing_mask.any().any():
            # Prepare data for imputation
            imputation_data = data_imputed[predictors + level_questions]

            # Perform imputation
            data_imputed_level = impute_data(imputer, imputation_data, level_questions)

            # Update data_imputed with imputed values
            data_imputed[level_questions] = data_imputed_level[level_questions]
        else:
            print(f"No missing values to impute at level {level}")

    return data_imputed


def data_integrity_check(data_missing, data_imputed):
    """
    Checks that all non-missing values in data_missing are the same in data_imputed.

    Parameters:
    - data_missing: DataFrame with missing values.
    - data_imputed: DataFrame after imputation.

    Returns:
    - discrepancies: DataFrame containing positions where values differ.
    - columns_match: Boolean indicating whether column names and orders match.
    """
    # Ensure columns are in the same order
    # data_missing = data_missing[data_imputed.columns]

    # Check if column names and orders match
    columns_match = list(data_missing.columns) == list(data_imputed.columns)
    assert columns_match == True

    # Create a mask where data_missing is not NaN
    non_missing_mask = ~data_missing.isnull()

    # Compare values at positions where data_missing is not NaN
    differences = (data_missing != data_imputed) & non_missing_mask

    # It is possible to have a difference here
    # Because you can have a super-question that is answered
    # both "Yes" and "No" (which we code as missing)
    # and then a sub-question answered "Yes".
    # If we impute the super-question as "No" then the sub-question
    # will be imputed as "No"--and this will seem like a
    # discrepancy against the original data.
    # because of this, we allow for few discrepancies
    # Should be on the order of maximum 10.
    differences_count = differences.eq(True).sum().sum()
    assert differences_count < 10


def hierarchical_relationship_check(data_imputed, df_relationships):
    """
    Checks that child questions have a value of 0 whenever their parent question has a value of 0.

    Parameters:
    - data_imputed: DataFrame after imputation.
    - df_relationships: DataFrame containing parent-child relationships.

    Returns:
    - violations: List of dictionaries detailing any violations found.
    """
    violations = []

    # Ensure that the necessary columns are in df_relationships
    if not {"question_id", "parent_question_id"}.issubset(df_relationships.columns):
        raise ValueError(
            "df_relationships must contain 'question_id' and 'parent_question_id' columns."
        )

    # Iterate over all parent-child relationships
    for _, row in df_relationships.iterrows():
        child = row["question_id"]
        parent = row["parent_question_id"]

        # Check if both parent and child columns exist in data_imputed
        if parent in data_imputed.columns and child in data_imputed.columns:
            # Create a mask where parent is 0
            parent_zero_mask = data_imputed[parent] == 0

            # Check if child is not 0 where parent is 0
            violation_mask = parent_zero_mask & (data_imputed[child] != 0)

            if violation_mask.any():
                violation_indices = data_imputed[violation_mask].index.tolist()
                violations.append(
                    {"parent": parent, "child": child, "indices": violation_indices}
                )

    assert len(violations) == 0
