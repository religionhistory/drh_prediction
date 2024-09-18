import pandas as pd
import numpy as np


def impute_data(imputer, data_missing, imputation_columns):
    """
    Performs imputation on the given data using the provided imputer.
    """
    # Perform imputation
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)

    # Handle non-binary outputs (e.g., from regressors)
    data_imputed[imputation_columns] = data_imputed[imputation_columns].apply(
        lambda x: 1 if x > 0.5 else 0
    )

    return data_imputed


from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef


def evaluate_imputation(data_imputed, data_original, imputation_columns):
    """
    Evaluates the imputed data against the original data using F1 score, accuracy, and MCC.
    """
    # Identify where missing values were introduced
    mask = data_imputed[imputation_columns].isnull()
    true_values = data_original[imputation_columns][mask]
    imputed_values = data_imputed[imputation_columns][mask]

    metrics = {"f1_score": [], "accuracy": [], "mcc": []}
    metrics_dict = {col: {} for col in imputation_columns}

    for column in imputation_columns:
        y_true = true_values[column].dropna()
        y_pred = imputed_values[column].loc[y_true.index]

        if y_true.empty or y_true.nunique() < 2 or y_pred.nunique() < 2:
            f1 = np.nan
            acc = np.nan
            mcc = np.nan
        else:
            f1 = f1_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)

        metrics["f1_score"].append(f1)
        metrics["accuracy"].append(acc)
        metrics["mcc"].append(mcc)

        metrics_dict[column]["f1_score"] = f1
        metrics_dict[column]["accuracy"] = acc
        metrics_dict[column]["mcc"] = mcc

    # Calculate average metrics
    avg_metrics = {metric: np.nanmean(values) for metric, values in metrics.items()}

    return avg_metrics, metrics_dict
