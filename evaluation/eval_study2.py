import pandas as pd
import warnings
from evaluation_functions import basic_metrics_study2
from constants import method_grid

warnings.filterwarnings("ignore")  # this makes sense here for "precision" warning

# prepare function inputs
nan_path = f"../imputation/output/study2/additional_NA"
data_path = "../imputation/output/study2"
df_complete = pd.read_csv("../data/preprocessed/answers_study2.csv")

# run metrics
df_metrics = basic_metrics_study2(method_grid, df_complete, nan_path, data_path)
df_metrics.to_csv("evaluation/metrics_study2.csv", index=False)
