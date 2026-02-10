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
# Import requirements from PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
# ---------------------------------------- #

# ---------------------------------------- #
# Reformat data to a torch-friendly format, and set the batch size
# (ie, how many points to train on at once)
batch_size = 10
x_tr_tensor = torch.from_numpy(x_train.to_numpy()).to(torch.float32)
# 'Normalize' data
x_tr_tensor = F.normalize(x_tr_tensor, p=2.0, dim=1)
y_tr_tensor = torch.from_numpy(y_train.to_numpy()).to(torch.float32).unsqueeze(1)

train_dataset = TensorDataset(x_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# ---------------------------------------- #

# ---------------------------------------- #
# Create a 3-layer neural network with a "ReLU" activation function
# ReLU is a "rectified linear unit". There are other
# popular functions, such as sigmoid 
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(30, 48) # Input layer (30 features) to a hidden layer (48 nodes)
        # self.relu = nn.ReLU()
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(48, 1)  # Hidden layer (48 nodes) to output layer (1 output)

    # The model is responsible for handling the "forward pass"
    # training
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
# ---------------------------------------- #

# ---------------------------------------- #
# Set up the loss function (penalty for incorrect labels)
criterion = nn.BCEWithLogitsLoss() # For binary classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.01)
# ---------------------------------------- #

# ---------------------------------------- #
num_epochs = 100

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # 1. Clear residual gradients from the previous iteration
        optimizer.zero_grad()

        # 2. Make a Forward Pass and get the output (predictions)
        outputs = model(inputs)

        # 3. Calculate the loss
        loss = criterion(outputs, labels)

        # 4. Perform a Backward pass to calculate gradients
        loss.backward()

        # 5. Run the optimizer to update the weights
        optimizer.step()

    # Print loss every 10 epochs (optional)
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Finished Training')
# ---------------------------------------- #

# ---------------------------------------- #
# Convert test data to torch-friendly format
x_test_tensor = torch.from_numpy(x_test.to_numpy()).to(torch.float32)
# 'Normalize' data
x_test_tensor = F.normalize(x_test_tensor, p=2.0, dim=1)
y_test_tensor = torch.from_numpy(y_test.to_numpy()).to(torch.float32).unsqueeze(1)

train_dataset = TensorDataset(x_test_tensor, y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# ---------------------------------------- #

# ---------------------------------------- #
# Move the model to "eval" mode
# This changes internal state to avoid accidental training
model.eval()
# Get predictions from the test set
# First, convert from model outputs (-inf-inf range)
# to probabilities
label_predicted_probabilities = F.sigmoid(model(x_test_tensor))
# Convert probabilities to either 0 or 1, here cutting off at 0.5
predicted_labels = (label_predicted_probabilities >= 0.5).int()
# ---------------------------------------- #

# ---------------------------------------- #
# Compute accuracy where the label matches ground truth
correct_predictions = (predicted_labels == y_test_tensor).float()
accuracy = correct_predictions.mean().item()
print(f"Accuracy: {accuracy * 100:.2f}%")
# ---------------------------------------- #
