# ---------------------------------------- #
## Notebooks can be used to run sections of script
## and keep the results in memory

## Scripts, on the other hand, run once and forget everything
# ---------------------------------------- #

# ---------------------------------------- #
# Basic imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
# ---------------------------------------- #

# ---------------------------------------- #
# Data shared among models #
# Data loading
data = sklearn.datasets.load_breast_cancer(as_frame=True)
data_as_DataFrame = data.frame
# Alias to something easier to work with
df = data_as_DataFrame
# ---------------------------------------- #

# ---------------------------------------- #
# Local requirements
from sklearn.decomposition import PCA
# ---------------------------------------- #

# ---------------------------------------- #
# Construct the PCA object and perform fit to adjust object internal state
pca = PCA(n_components=3)
pca.fit(data.data)

data_reduced = PCA(n_components=3).fit_transform(data.data)

# Introspect into the results of the calculations
print(f"Explained variance: {pca.explained_variance_}")
print(f"Principal Components:\n{pca.components_}")
# ---------------------------------------- #

# ---------------------------------------- #
# ---------------------------------------- #
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

scatter = ax.scatter(
    data_reduced[:, 0],
    data_reduced[:, 1],
    data_reduced[:, 2],
    c=data.target,
    alpha=0.6
)

ax.set(
    title="First three principal components",
    xlabel="1st Principal Component",
    ylabel="2nd Principal Component",
    zlabel="3rd Principal Component",
)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

# Add a legend
legend1 = ax.legend(
    scatter.legend_elements()[0],
    data.target_names.tolist(),
    loc="upper right",
    title="Cancer class",
)
ax.add_artist(legend1)
# The command line isn't able to view images at the terminal
plt.savefig("principal_components_analysis.png")
# ---------------------------------------- #
