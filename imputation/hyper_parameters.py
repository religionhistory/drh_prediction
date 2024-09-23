param_grid_rf = {
    "n_estimators": [20, 50, 100],  # default: 100
    "max_depth": [None],  # default: None
    "max_features": ["sqrt"],  # default: "sqrt"
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
    "n_estimators": [20, 50, 100],
    "max_depth": [None],  # complexity
    "learning_rate": [0.1],
    "subsample": [0.9, 1.0],  # no subsampling
    "colsample_bytree": [0.9, 1.0],  # no subsampling
    "objective": ["reg:squarederror"],
}

# bayesian ridge as well
param_grid_bayesian_ridge = {
    "alpha_1": [1e-6, 1e-5],
    "lambda_1": [1e-6, 1e-5],
    "fit_intercept": [True, False],
}
