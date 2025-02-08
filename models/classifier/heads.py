import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import torch.nn.functional as F

import torch.nn as nn


import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=10, dropout=0.5):
        super(ClassificationHead, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        self.layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout2 = nn.Dropout(dropout)

        self.layer3 = nn.Linear(hidden_size//2, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        self.dropout3 = nn.Dropout(dropout)

        self.linear4 = nn.Linear(hidden_size//2, output_size)  # No activation here!

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout3(x)

        x = self.linear4(x)  # No activation here (CrossEntropyLoss applies softmax)
        return x

class AttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        att_w = F.softmax(self.W(batch_rep).squeeze(-1), dim=1).unsqueeze(-1)  # dim=1 is the sequence length dimension
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class ClsAttn(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, output_size=2, dropout=0.5):
        super(ClsAttn, self).__init__()
        self.attn_pooling = AttentionPooling(input_size)
        self.FF = ClassificationHead(input_size, hidden_size, output_size, dropout)

    def forward(self, x):
        x = self.attn_pooling(x)
        x = self.FF(x)
        return x

    import torch
    import torch.nn.functional as F

    def inference(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)  # Forward pass
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax

            # Get the class with the maximum probability
            predicted_min = torch.argmax(probabilities, dim=1)

            # Create a tensor with True where probabilities exceed the threshold
            predicted_classes = (probabilities > threshold).int()

            # Ensure that the class with the max probability is included, even if its prob is below the threshold
            for i in range(probabilities.size(0)):  # Loop over batch
                predicted_classes[i, predicted_min[i]] = 1

            predicted_classes = predicted_classes.cpu().numpy()

        return predicted_classes