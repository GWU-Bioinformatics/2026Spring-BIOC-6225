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
# Separate data based on label
malignant_x_points = df['mean perimeter'][df['target'] == 0]
malignant_y_points = df['mean area'][df['target'] == 0]
benign_x_points = df['mean perimeter'][df['target'] == 1]
benign_y_points = df['mean area'][df['target'] == 1]
labels = df['target']
legend_text = [str(target_name) for target_name in list(data.target_names)]
# ---------------------------------------- #
# ---------------------------------------- #
# Fit a Linear Regression
LR_benign = sklearn.linear_model.LinearRegression()
benign_model = LR_benign.fit(benign_x_points.to_frame(), benign_y_points.to_frame())
benign_line = benign_model.predict(benign_x_points.to_frame())

LR_malignant = sklearn.linear_model.LinearRegression()
malignant_model = LR_malignant.fit(malignant_x_points.to_frame(), malignant_y_points.to_frame())
malignant_line = malignant_model.predict(malignant_x_points.to_frame())
# ---------------------------------------- #
# ---------------------------------------- #
# Create a plot colored as a function of benign/malignant label
plt.scatter(malignant_x_points, malignant_y_points, c='purple', label="Malignant", alpha=0.2)
plt.scatter(benign_x_points, benign_y_points, c='yellow', label="Benign", alpha=0.2)
plt.plot(malignant_x_points, malignant_line, color='black', linewidth=1, label='Malignant model', alpha=0.7)
plt.plot(benign_x_points, benign_line, color='blue', linewidth=1, label='Benign model', alpha=0.7)
plt.xlabel('mean perimeter (mm)')
plt.ylabel('mean area (mm)')
plt.legend()
# The command line isn't able to view images at the terminal
plt.savefig("linear_regression.png")
# ---------------------------------------- #
