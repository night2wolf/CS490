# Import necessary packages
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import helper
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

###
# Create the network and look at it's text representation
model = Network()
model
# Your solution here


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        # First hidden layer, 10 units - one for each digit
        self.fc2 = nn.Linear(128, 64)
        # Second Hidden layer
        self.fc3 = nn.Linear(64, 10)

        # Define sigmoid activation and softmax output
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x
