import torch
import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from itertools import cycle

from time import time
from numpy import arange
from sklearn.model_selection import train_test_split
from copy import deepcopy


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
            input_copy[input_copy.index(x)] = nn.LazyConv2d(x[1], x[2])
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

    Parameters:
        func (callable): The function to generate data from (e.g., lambda x: x**2).
        test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).

    Returns:
        X_train, X_test, y_train, y_test: Training and test data.
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

    Parameters:
        model (nn.Module): Neural network model.
        X_train (np.ndarray/torch.Tensor): Training input data (shape: [num_samples, 1]).
        y_train (np.ndarray/torch.Tensor): Training target data (shape: [num_samples, 1]).
        X_test (np.ndarray/torch.Tensor): Test input data (shape: [num_samples, 1]).
        y_test (np.ndarray/torch.Tensor): Test target data (shape: [num_samples, 1]).
        time_limit (int): Training duration in seconds (default: 10).
        batch_size (int): Batch size for training (default: 32).
        learning_rate (float): Learning rate for the optimizer (default: 0.001).

    Returns:
        total_inaccuracy (float): Sum of absolute differences between predictions and true values.
        std_inaccuracy (float): Standard deviation of the inaccuracy.
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
        # Get next batch
        batch_X, batch_y = next(train_loader_infinite)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
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

def get_custom_mnist(
    num_0: tuple = None,  # (train_samples, test_samples)
    num_1: tuple = None,
    num_2: tuple = None,
    num_3: tuple = None,
    num_4: tuple = None,
    num_5: tuple = None,
    num_6: tuple = None,
    num_7: tuple = None,
    num_8: tuple = None,
    num_9: tuple = None,
):
    """
    Create custom MNIST datasets with specified numbers of samples per digit.
    
    Args:
        num_0 to num_9: Tuples specifying (training_samples, test_samples) for each digit.
                        Omit or set to None to exclude a digit.
    
    Returns:
        (X_train, y_train), (X_test, y_test): Tuple of training and test datasets
        with images sized 28x28 and normalized to [0, 1]
    """
    # Load full MNIST datasets
    train_set = datasets.MNIST(root='./data', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
    
    test_set = datasets.MNIST(root='./data', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    # Create storage tensors
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Process each digit specification
    for digit in range(10):
        spec = locals()[f'num_{digit}']
        if spec is None:
            continue
            
        train_samples, test_samples = spec
        
        # Get training data for digit
        train_idx = (train_set.targets == digit).nonzero().squeeze()
        if len(train_idx) < train_samples:
            raise ValueError(f"Digit {digit} only has {len(train_idx)} training samples")
            
        X_train.append(train_set.data[train_idx[:train_samples]])
        y_train.append(torch.full((train_samples,), digit))
        
        # Get test data for digit
        test_idx = (test_set.targets == digit).nonzero().squeeze()
        if len(test_idx) < test_samples:
            raise ValueError(f"Digit {digit} only has {len(test_idx)} test samples")
            
        X_test.append(test_set.data[test_idx[:test_samples]])
        y_test.append(torch.full((test_samples,), digit))

    # Combine all selected samples
    X_train = torch.cat(X_train) if X_train else torch.tensor([])
    y_train = torch.cat(y_train) if y_train else torch.tensor([])
    X_test = torch.cat(X_test) if X_test else torch.tensor([])
    y_test = torch.cat(y_test) if y_test else torch.tensor([])

    # Normalize and add channel dimension
    X_train = X_train.float().div(255).unsqueeze(1)  # Shape: [N, 1, 28, 28]
    X_test = X_test.float().div(255).unsqueeze(1)

    return (X_train, y_train), (X_test, y_test)

def train_classification_model_time_based(
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
    Train a classification neural network for a specified time and return test inaccuracy percentage.

    Parameters:
        model (nn.Module): Classification neural network
        X_train (np.ndarray/torch.Tensor): Training input data (shape: [num_samples, features])
        y_train (np.ndarray/torch.Tensor): Training labels (shape: [num_samples])
        X_test (np.ndarray/torch.Tensor): Test input data
        y_test (np.ndarray/torch.Tensor): Test labels
        time_limit (int): Training duration in seconds (default: 10)
        batch_size (int): Batch size for training (default: 32)
        learning_rate (float): Learning rate for optimizer (default: 0.001)

    Returns:
        inaccuracy_percent (float): Percentage of incorrect predictions (100% - accuracy)
    """
    # Move model and data to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert data to tensors and ensure correct shapes
    if not isinstance(X_train, torch.Tensor):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)

    # Ensure input data has shape (num_samples, features)
    if X_train.dim() == 1:
        X_train = X_train.unsqueeze(1)
    if X_test.dim() == 1:
        X_test = X_test.unsqueeze(1)

    # Move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Create DataLoader with infinite iteration
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_infinite = cycle(train_loader)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Time-based training loop
    model.train()
    start_time = time()
    while time() - start_time < time_limit:
        # Get next batch
        inputs, labels = next(train_loader_infinite)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # Predict in batches to handle large datasets
        for inputs, labels in DataLoader(TensorDataset(X_test, y_test), batch_size=512):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    inaccuracy_percent = (1 - accuracy) * 100

    return inaccuracy_percent

model = [["Conv", 1, 8, 3], ["ReLU"], ["Pool", 2], ["Conv", 8, 16, 3], ["ReLU"], ["Pool", 2], ["Flatten"], 
         ["Linear", 50], ["ReLU"], ["Linear", 10]]
model = makemodel(model)