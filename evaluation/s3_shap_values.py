import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
import json
from constants import short_labels

# setup
df = pd.read_csv("../data/preprocessed/answers_study2.csv")
question_columns = df.filter(regex="Q_").columns
question_columns = list(question_columns)
other_predictors = df.columns.difference(question_columns)
other_predictors = other_predictors.drop("entry_id")
other_predictors = list(other_predictors)

# load best settings for rf
best_model = "IterativeImputer_RandomForest"
with open(f"../data/study2/hyperparams/best_params_{best_model}.json", "r") as f:
    best_params = json.load(f)

# here do only super-questions
# otherwise it gets really weird
question_levels = pd.read_csv("../data/preprocessed/question_level_study2.csv")
question_levels = question_levels[question_levels["question_level"] == 1]
top_questions = question_levels["question_id"].tolist()
top_questions = [f"Q_{q}" for q in top_questions]
question_columns = [q for q in question_columns if q in top_questions]
assert top_questions == question_columns

"""
First get the overall strongest questions.
Generally temporal span is a strong predictor + 
Questions from "Society and Institutions". 
I am not sure whether it is just because there are many
questions from this category that internally have strong
predictive power, or whether it is because these questions
are really good predictors across other categories. 
"""

# List to store results
results = []

# Loop through each question column
for target_question in question_columns:
    print(f"Processing: {target_question}")

    # Define predictors: all other questions + additional variables
    predictors = other_predictors + [
        col for col in question_columns if col != target_question
    ]

    # Train data where the target question is not missing
    training_data = df.dropna(subset=[target_question])
    X = training_data[predictors]
    y = training_data[target_question]

    # Train the RandomForest model
    clf = RandomForestClassifier(**best_params.get(best_model, {}), random_state=0)
    clf.fit(X, y)

    # Calculate SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(clf)
    shap_values = np.array(explainer.shap_values(X))

    # Extract SHAP values for class 1 (positive class)
    shap_values_class1 = shap_values[:, :, 1]

    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.mean(np.abs(shap_values_class1), axis=0)

    # Store results in the list
    for predictor, shap_value in zip(predictors, mean_abs_shap):
        results.append(
            {
                "outcome": target_question,
                "predictor": predictor,
                "mean_absolute_shap": shap_value,
            }
        )

# Convert results to a DataFrame
feature_importance_df = pd.DataFrame(results)

# Display the feature importance DataFrame
best_features_overall = (
    feature_importance_df.groupby("predictor")["mean_absolute_shap"]
    .mean()
    .reset_index(name="mean_absolute_shap")
    .sort_values(by="mean_absolute_shap", ascending=False)
    .head(10)
)

best_features_overall = best_features_overall.rename(
    columns={"predictor": "predictor_code"}
)
best_features_overall["predictor_name"] = best_features_overall["predictor_code"].map(
    short_labels
)
best_features_overall = best_features_overall[["predictor_name", "mean_absolute_shap"]]

# to latex
best_features_overall.to_latex("../tables/s3_best_features_overall.tex", index=False)

"""
Pick a single question to see which predictors are best. 
Many things could be interesting here, but let us try with
"Belief in afterlife". 
"""

# find the question
question_names = pd.read_csv("../data/preprocessed/answers.csv")
target_question = "Q_2900"  # Belief in afterlife

# Train data where the target question is not missing
training_data = df.dropna(subset=target_question)
predictors = other_predictors + [
    col for col in question_columns if col != target_question
]
X = training_data[predictors]
y = training_data[target_question]

clf = RandomForestClassifier(**best_params.get(best_model, {}), random_state=0)
clf.fit(X, y)

# Calculate SHAP values using TreeExplainer
explainer = shap.TreeExplainer(clf)
shap_values = np.array(explainer.shap_values(X))

# predicting the positive class
shap_values_class1 = shap_values[:, :, 1]
feature_names = [short_labels.get(col, col) for col in X.columns]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
shap.summary_plot(shap_values_class1, X, feature_names=feature_names, show=False)
plt.savefig(
    "../figures/study3/s3_shap_values_belief_in_afterlife.pdf", bbox_inches="tight"
)
plt.close()
