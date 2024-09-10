import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# are there some where we cannot calculate correlations #
df = pd.read_csv("temporary.csv")
dfX = df.filter(regex="^X")

# calculate correlations
correlation_mat = dfX.corr()

# plot
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_mat, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("")
plt.show()
