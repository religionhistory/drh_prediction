""" 
to-do: 
make a general pipeline that makes sense.
do that for the random forest classifier.
then try to figure out what actually works.
would be cool to try XGBoost as well. 
not sure about hyper parameter tuning. 
"""

import numpy as np
import pandas as pd

# start simple
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("../data/study1/additional_NA/MAR_0.2.csv")
question_columns = df.filter(regex="Q_").columns

original_data = df[df.columns.difference(df.filter(regex="Q_").columns)]
imputer = IterativeImputer(
    estimator=RandomForestClassifier(), max_iter=10, random_state=0
)
df_imputed = imputer.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed, columns=df.columns)
df_imputed[original_data.columns] = original_data

df_imputed

### getting shap values ###
### this should not be too different from models ###
imputed_results = []
n_iterations = 10
for i in range(n_iterations):
    # Set up the imputer without a fixed random state
    imputer = IterativeImputer(
        estimator=RandomForestClassifier(), max_iter=10, random_state=None
    )

    # Fit and transform the whole DataFrame
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Append only the imputed question columns
    imputed_results.append(df_imputed[question_columns].values)

# Convert results to a NumPy array for easier computation
imputed_array = np.array(
    imputed_results
)  # Shape: (n_iterations, n_samples, n_features)

# Calculate mean and standard deviation for each imputed value across the iterations
imputed_means = np.nanmean(imputed_array, axis=0)
imputed_stds = np.nanstd(imputed_array, axis=0)

# Create a DataFrame to show imputed means and uncertainty (stds)
imputed_summary = pd.DataFrame(imputed_means, columns=question_columns)
imputed_uncertainty = pd.DataFrame(imputed_stds, columns=question_columns)

imputed_summary
imputed_uncertainty
