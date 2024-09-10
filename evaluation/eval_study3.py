import pandas as pd
import warnings

warnings.filterwarnings("ignore")  # this makes sense here for "precision" warning
from evaluation_functions import flag_constant_variables, pairwise_correlations_study3
from constants import method_grid

# currently only mice
method_grid = [x for x in method_grid if "mice" in x]

# paths to nan files
data_path = "../imputation/output/study3"
df_complete = pd.read_csv("../data/preprocessed/answers_study1.csv")

### calculate pairwise metrics ###
constant_columns = flag_constant_variables(method_grid, df_complete, data_path)

df_pairwise_var = pairwise_correlations_study3(
    method_grid, df_complete, data_path, constant_columns
)

df_pairwise_var.to_csv("evaluation/correlations_study3.csv", index=False)
