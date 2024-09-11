import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

# setup
df = pd.read_csv("../data/preprocessed/answers_study2.csv")
question_columns = df.filter(regex="Q_").columns
question_columns = list(question_columns)
other_predictors = df.columns.difference(question_columns)
other_predictors = other_predictors.drop("entry_id")
other_predictors = list(other_predictors)

"""
For just a single question on the small dataset. 
"""

target_question = "Q_3166"
training_data = df.dropna(subset=target_question)
predictors = other_predictors + [
    col for col in question_columns if col != target_question
]
X = training_data[predictors]
y = training_data[target_question]

clf = RandomForestClassifier(n_estimators=100, max_depth=3)
clf.fit(X, y)

# Calculate SHAP values using TreeExplainer
explainer = shap.TreeExplainer(clf)
shap_values = np.array(explainer.shap_values(X))

# predicting the positive class
shap_values_class1 = shap_values[:, :, 1]
shap.summary_plot(shap_values_class1, X, feature_names=X.columns)

### for understanding ###
answers = pd.read_csv("../data/preprocessed/answers.csv")
pd.set_option("display.max_colwidth", None)
answers[answers["question_id"] == 3166].head(1)

""" Example: 
"Does the religious group in question possess its own distinct written language?"

Interpretation: 
"Yes" increased when: 
- Religious group provides public food storage (3149), 
- There is special corpse treatments (2876),
- There are different types of monuments (2322),
- There are monuments (2265)
- Region is South Asia

"No" increased when: 
- Religion is recent (year from and year to is high),
- Region is Africa
- There is a supreme high god (2919)
"""

""" 
Overall approach: 

Would maybe be cool to have 2 of these plots side by side. 
And then a table / heatmap of just how strong the predictors are overall. 
Is there a way to quantify this?

Of course we could even go further down and highlight a specific case.
For instance "For this weird religion we have prediction X, why?" 
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
    clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
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
feature_importance_df.groupby("predictor")["mean_absolute_shap"].mean().sort_values(
    ascending=False
).head(10)

""" 
Best predictors tend to be questions and date variables.

Q 3218: permissible to worship beings other than the high god
Q 3167: water management provided by institution other than group
Q 3154: interact with institutionalized judicial system other than group
Q 3135: subject to formal legal code other than group
Q 3162: ...

These might all be selected because they are predicting each other...
Not clear that these are really what tell us about the predictions. 
"""
