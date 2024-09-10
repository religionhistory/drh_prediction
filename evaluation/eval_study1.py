import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # this makes sense here for "precision" warning
from evaluation_functions import (
    basic_metrics_study1,
    pairwise_correlations_study1,
    flag_constant_variables,
)
from constants import method_grid

# paths to nan files
nan_path = f"../imputation/output/study1/additional_NA"
data_path = "../imputation/output/study1"
df_complete = pd.read_csv("../data/preprocessed/answers_study1.csv")

### calculate basic metrics ###
df_metrics = basic_metrics_study1(method_grid, df_complete, nan_path, data_path)
df_metrics.to_csv("evaluation/metrics_study1.csv", index=False)

### calculate pairwise metrics ###
"""
So we are basically saying that if any column in constant for any
imputation method (should be often for "mode") then we are not 
considering the correlation with this variable and any other variable.
"""

# for this we will want to extend the method grid #
method_grid = method_grid + ["additional_NA"]
constant_columns = flag_constant_variables(method_grid, df_complete, data_path)
df_pairwise_var = pairwise_correlations_study1(
    method_grid, df_complete, data_path, constant_columns
)
df_pairwise_var.to_csv("evaluation/correlations_study1.csv", index=False)
