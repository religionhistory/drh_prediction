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
import numpy as np
import pandas as pd
import os
import re

# start simple
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
import xgboost

# load full data
data_full = pd.read_csv("../data/preprocessed/answers_study1.csv")
question_columns = data_full.filter(regex="Q_").columns
original_data = data_full[
    data_full.columns.difference(data_full.filter(regex="Q_").columns)
]
n_samples, n_features = data_full.shape

# prep
files_missingness = os.listdir("../data/study1/additional_NA")
pattern = r"NA_(MCAR|MNAR|MAR)_(\d+)_(\d+).csv"
imputer = SimpleImputer(strategy="most_frequent")

# run all files
for file_missing in files_missingness:
    missing_type = re.match(pattern, file_missing).groups()
    data_missing = pd.read_csv(f"../data/study1/additional_NA/{file_missing}")
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)
    data_imputed[original_data.columns] = original_data
    data_imputed.to_csv(f"../data/study1/mode/{file_missing}", index=False)

# where do I tune hyperparameters for this?
# Note that although KNNImputer and KNeighborsRegressor are different
# We have tested this and gotten essentially identical results.
imputer = KNNImputer(n_neighbors=10)
for file_missing in files_missingness:
    missing_type = re.match(pattern, file_missing).groups()
    data_missing = pd.read_csv(f"../data/study1/additional_NA/{file_missing}")
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)
    data_imputed[original_data.columns] = original_data
    data_imputed.to_csv(f"../data/study1/knn_eval/{file_missing}", index=False)
    # now get the actual prediction
    data_imputed = data_imputed.round().astype(int)
    data_imputed.to_csv(f"../data/study1/knn/{file_missing}", index=False)

# Do we need hyper-params here?
# Also not 100% sure that this makes sense
# Will occasionally give a prediction outside of 1/0.
imputer = IterativeImputer(BayesianRidge(), max_iter=25, random_state=0, tol=1e-3)
for file_missing in files_missingness:
    missing_type = re.match(pattern, file_missing).groups()
    data_missing = pd.read_csv(f"../data/study1/additional_NA/{file_missing}")
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)
    data_imputed[original_data.columns] = original_data
    data_imputed.to_csv(f"../data/study1/ridge_eval/{file_missing}", index=False)
    # now get the actual prediction
    data_imputed = data_imputed.map(lambda x: 1 if x > 0.5 else 0)
    data_imputed.to_csv(f"../data/study1/ridge/{file_missing}", index=False)

# Here we should do hyper-parameter tuning as well.
# This is a lot slower than the other methods so far.
imputer = IterativeImputer(
    estimator=RandomForestClassifier(
        n_estimators=100,  # default
        # max_depth=10,  # default no limit
        bootstrap=True,  # default
        # other things to consider ... (e.g., max samples for bootstrap)
    ),
    max_iter=25,
    random_state=0,
    tol=1e-3,
)
for file_missing in files_missingness:
    print(file_missing)
    missing_type = re.match(pattern, file_missing).groups()
    data_missing = pd.read_csv(f"../data/study1/additional_NA/{file_missing}")
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)
    data_imputed[original_data.columns] = original_data
    data_imputed.to_csv(f"../data/study1/rf/{file_missing}", index=False)

# can we get XGBoost to work for this?
# There is also max depth, max leaves, max bin, grow policy, learning rate, etc.
imputer = IterativeImputer(
    estimator=xgboost.XGBRegressor(  # should probably be XGBClassifier
        n_estimators=100,  # but should be tuned
        tree_method="hist",  # fastests, but see "approx"
    ),
    max_iter=25,
    random_state=0,
    tol=1e-3,
)
for file_missing in files_missingness:
    print(file_missing)
    missing_type = re.match(pattern, file_missing).groups()
    data_missing = pd.read_csv(f"../data/study1/additional_NA/{file_missing}")
    data_imputed = imputer.fit_transform(data_missing)
    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)
    data_imputed[original_data.columns] = original_data
    data_imputed.to_csv(f"../data/study1/xgb_eval/{file_missing}", index=False)
    # now get the actual prediction
    data_imputed = data_imputed.map(lambda x: 1 if x > 0.5 else 0)
    data_imputed.to_csv(f"../data/study1/xgb/{file_missing}", index=False)

# get XGBClassifier() to work 
# This seems not as easy as one might think.
# Errors on the non-binary columns. 
# Will have to consider whether this would even improve 
# on the other XGB approach. 
imputer = IterativeImputer(
    estimator=xgboost.XGBClassifier(
        n_estimators=100,
        tree_method="hist",
    ),
    max_iter=25,
    random_state=0,
    tol=1e-3,
)
# Separate into binary and non-binary
data_to_impute = data_missing[question_columns]
predictor_columns = data_full.columns.difference(data_full.filter(regex="Q_").columns)

for file_missing in files_missingness:
    print(file_missing)
    missing_type = re.match(pattern, file_missing).groups()
    data_missing = pd.read_csv(f"../data/study1/additional_NA/{file_missing}")
    # adding a step here
    data_to_impute = data_missing[question_columns]
    data_to_predict = data_missing[predictor_columns]
    data_for_imputation = 
    #
    data_imputed = imputer.fit_transform(data_missing)

    data_imputed = pd.DataFrame(data_imputed, columns=data_missing.columns)
    data_imputed[original_data.columns] = original_data
    data_imputed.to_csv(f"../data/study1/xgb_class/{file_missing}", index=False)
    # now get the actual prediction
    # data_imputed = data_imputed.map(lambda x: 1 if x > 0.5 else 0)
    # data_imputed.to_csv(f"../data/study1/xgbclass/{file_missing}", index=False)
data_missing
