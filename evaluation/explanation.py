from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from shap import TreeExplainer
from shap import summary_plot
import pandas as pd
import numpy as np

# setup
y_variable = "X4828"

# load actual data set
answers_s1 = pd.read_csv("../data/preprocessed/answers_study1.csv")

# drop entry ID
answers_s1 = answers_s1.drop("entry_id", axis=1)

# now split into X and Y
y = answers_s1[[y_variable]]
X = answers_s1.drop(y_variable, axis=1)

y = y.values.ravel()
X = X.values

# now split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Random Forest Classification
clf = RandomForestClassifier(n_estimators=100, max_depth=3)
clf.fit(X_train, y_train)
explainer = TreeExplainer(clf)
shap_values = np.array(explainer.shap_values(X_train))
print(shap_values.shape)  # (2, 261, 27)

# This is a sanity check
shap_values_ = shap_values.transpose((1, 0, 2))
np.allclose(clf.predict_proba(X_train), shap_values_.sum(2) + explainer.expected_value)

# Now we can plot
# No need to be concerned about shap_values[1] that is just flipped.
summary_plot(shap_values[0], X_train)

"""
Interpretation: 
- From the top are the most important. 
- Feature 5: lower values tend to result in more "Yes"
- Feature 3, 5: higher values tend to result in more "Yes" 
"""

### figure out what the labels correspond to ###
df_interpret = pd.DataFrame(answers_s1.columns, columns=["feature"])
df_interpret["index"] = df_interpret.index

"""
5 = east asia
3 = region_africa
15 = X4729 (find out what this is)
8 = region_oceania_australia
1 = year_to
"""

# check the regions first
answers_s1.groupby("region_east_asia")[y_variable].mean()  # makes sense
answers_s1.groupby("region_africa")[y_variable].mean()  # makes sense
answers_s1.groupby("X4729")[y_variable].mean()  # makes sense
answers_s1.groupby("region_oceania_australia")[y_variable].mean()  # makes sense
