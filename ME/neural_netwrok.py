# neural_network.py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from itertools import cycle
from time import time
from numpy import arange
from sklearn.model_selection import train_test_split
from copy import deepcopy
import random
import math
import numpy as np

def makemodel(input_list):
    input_copy = deepcopy(input_list)
    
    # Turn input copy into a list of nn. functions
    for x in input_list:
        # Check the type of layer/activation specified in the first element of the sublist
        if x[0] == "Linear":
            input_copy[input_copy.index(x)] = nn.LazyLinear(x[1])
        elif x[0] == "Dropout":
            input_copy[input_copy.index(x)] = nn.Dropout()
        elif x[0] == "ReLU":
            input_copy[input_copy.index(x)] = nn.ReLU()
        elif x[0] == "LeakyReLU":
            input_copy[input_copy.index(x)] = nn.LeakyReLU()
        elif x[0] == "ELU":
            input_copy[input_copy.index(x)] = nn.ELU()
        elif x[0] == "Sigmoid":
            input_copy[input_copy.index(x)] = nn.Sigmoid()
        elif x[0] == "BatchNorm":
            input_copy[input_copy.index(x)] = nn.BatchNorm1d()
        elif x[0] == "Conv":
            input_copy[input_copy.index(x)] = nn.Conv2d(x[1], x[2], x[3])
        elif x[0] == "Pool":
            input_copy[input_copy.index(x)] = nn.MaxPool2d(x[1])
        elif x[0] == "Flatten":
            input_copy[input_copy.index(x)] = nn.Flatten()
    # Create Network
    model = nn.Sequential()
    for x in input_copy:
        model.append(x)
    return model

def generate_data(func, test_size=0.2, random_state=42):
    """
    Generate training and test data for a given function.
    """
    # Generate x values from -5 to 5 with a step size of 0.1
    x_values = arange(-5, 5.1, 0.1)
    # Compute y values using the given function
    y_values = func(x_values)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        x_values, y_values, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_and_test_model_time_based(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    time_limit=10,  # Training time in seconds
    batch_size=32,
    learning_rate=0.001,
):
    """
    Train a neural network for a specified time (in seconds) and evaluate its inaccuracy.
    Returns total and standard deviation of absolute inaccuracy.
    """
    # Move model and data to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert numpy arrays to tensors and ensure correct shape
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    # Ensure data has shape (num_samples, 1) for single-value I/O
    X_train = X_train.unsqueeze(1) if X_train.dim() == 1 else X_train
    y_train = y_train.unsqueeze(1) if y_train.dim() == 1 else y_train
    X_test = X_test.unsqueeze(1) if X_test.dim() == 1 else X_test
    y_test = y_test.unsqueeze(1) if y_test.dim() == 1 else y_test

    # Move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_infinite = cycle(train_loader)  # Infinite batch iterator

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop (time-based)
    model.train()
    start_time = time()
    while time() - start_time < time_limit:
        batch_X, batch_y = next(train_loader_infinite)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        inaccuracy = torch.abs(y_pred - y_test)
        total_inaccuracy = inaccuracy.sum().item()
        std_inaccuracy = inaccuracy.std().item()
    return total_inaccuracy, std_inaccuracy

# (Other functions such as get_custom_mnist and train_classification_model_time_based remain unchanged if needed)

# Sample model creation (this is just an example and not used by the Flask endpoint)
if __name__ == "__main__":
    model_config = [
        ["Conv", 1, 8, 3],
        ["ReLU"],
        ["Pool", 2],
        ["Conv", 8, 16, 3],
        ["ReLU"],
        ["Pool", 2],
        ["Flatten"],
        ["Linear", 50],
        ["ReLU"],
        ["Linear", 10]
    ]
    model = makemodel(model_config)
