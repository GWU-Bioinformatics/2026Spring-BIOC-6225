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
# Data loading
data = sklearn.datasets.load_breast_cancer(as_frame=True)
data_as_DataFrame = data.frame
# Alias to something easier to work with
df = data_as_DataFrame
# ---------------------------------------- #

# ---------------------------------------- #
# Re-sample the original data and perform a train(ing)/80%, test(ing)/20% data split
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# ---------------------------------------- #

# ---------------------------------------- #
# Local requirements
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
# ---------------------------------------- #

# ---------------------------------------- #
# Initialize the Naive Bayes object
gnb = GaussianNB()
# ---------------------------------------- #

# ---------------------------------------- #
# Fit the data with the resulting object
gnb.fit(x_train, y_train)
# Obtain predictions from the model
y_pred = gnb.predict(x_test)
# Print out the result
print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")
# ---------------------------------------- #

# ---------------------------------------- #

# ---------------------------------------- #

# ---------------------------------------- #

# ---------------------------------------- #
