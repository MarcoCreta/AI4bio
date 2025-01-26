import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import torch.nn.functional as F

import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=10, dropout=0.5):
        super(ClassificationHead, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # Added activation

        # Optional second layer
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(hidden_size, output_size)

        self.softmax = nn.Softmax(dim=0)
        # Removed Softmax

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        # If layer2 is used
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.softmax(x)
        return x
