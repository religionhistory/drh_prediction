import pandas as pd
import json
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor

# Import custom functions
from imputation_functions import (
    hierarchical_imputation,
    data_integrity_check,
    hierarchical_relationship_check,
)

# Load the original data
data_original = pd.read_csv(f"../data/preprocessed/answers_study2.csv")

# Load relationships and adjust question IDs
df_relationships = pd.read_csv(f"../data/preprocessed/question_level_study2.csv")
df_relationships["question_id"] = "Q_" + df_relationships["question_id"].astype(str)
df_relationships["parent_question_id"] = "Q_" + df_relationships[
    "parent_question_id"
].astype(str)

# Define imputation columns and non-question predictors
imputation_columns = data_original.filter(regex="^Q_").columns.tolist()
non_question_predictors = data_original.columns.difference(imputation_columns).tolist()

# locate the best imputer
imputer_name = "IterativeImputer_RandomForest"

# Load data
best_params = {}
with open(f"../data/study2/hyperparams/best_params_{imputer_name}.json", "r") as f:
    best_params[imputer_name] = json.load(f)

imputer = IterativeImputer(
    estimator=RandomForestClassifier(
        **best_params.get(imputer_name, {}), random_state=0
    ),
    max_iter=25,
    random_state=0,
    tol=1e-3,
)

# Hierarchical imputation
data_imputed = hierarchical_imputation(
    df_relationships=df_relationships,
    non_question_predictors=non_question_predictors,
    data_missing=data_original,
    imputer=imputer,
)

# Check data integrity
data_integrity_check = data_integrity_check(data_original, data_imputed)
hierarchical_relationship_check = hierarchical_relationship_check(
    data_imputed, df_relationships
)

# Save the imputed data
data_imputed.to_csv("../data/study3/imputed_data.csv", index=False)
