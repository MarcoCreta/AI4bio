import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
import math

class ClassificationHead(nn.Module):
    def __init__(self, input_size=256, hidden_size=128, output_size=10, dropout=0.5):
        super(ClassificationHead, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU()

        self.layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout2 = nn.Dropout(dropout)

        self.linear4 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        x = self.linear4(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim: int = 1280, max_len: int = 5):
        super(RotaryPositionalEncoding, self).__init__()
        self.dim = dim

        # Compute rotation frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_chunks=5, dim=1280)

        Returns:
            x with rotary positional embeddings applied.
        """
        batch_size, num_chunks, dim = x.shape  # (B, 5, 1280)
        device = x.device

        # Generate positions (chunk indices)
        positions = torch.arange(num_chunks, dtype=torch.float32, device=device).unsqueeze(1)  # (5, 1)
        freqs = torch.outer(positions.squeeze(), self.inv_freq)  # (5, dim // 2)

        # Compute sin and cos for RoPE transformation
        sin, cos = freqs.sin(), freqs.cos()
        sin, cos = sin.unsqueeze(0), cos.unsqueeze(0)  # Expand for batch size

        # Split embeddings into even and odd indices
        x_even, x_odd = x[..., 0::2], x[..., 1::2]

        # Apply RoPE rotation
        x_rotated = torch.cat([x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1)

        return x_rotated

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
    def __init__(self, input_size=1280, hidden_size=256, output_size=2, dropout=0.5, chunk_size=5):
        super(ClsAttn, self).__init__()
        if Config.TRAIN_ARGS['ATTN_POOLING']:
            if Config.TRAIN_ARGS['CHUNK_RE']:
                self.rotary_encoding = RotaryPositionalEncoding(input_size)
            if Config.TRAIN_ARGS['TOKEN_PE']:
                self.chunked_encoding = PositionalEncoding(input_size, dropout=dropout)
            self.attn_pooling = AttentionPooling(input_size)
        self.FF = ClassificationHead(input_size, hidden_size, output_size, dropout)

    def forward(self, x):
        if Config.TRAIN_ARGS['ATTN_POOLING']:
            if Config.TRAIN_ARGS['CHUNK_RE']:
                x = self.rotary_encoding(x)  # Apply RoPE first
            if Config.TRAIN_ARGS['TOKEN_PE']:
                x = self.chunked_encoding(x)  # Then absolute chunk encoding
            x = self.attn_pooling(x)  # Attention pooling

        x = self.FF(x)  # Classification head
        return x


    def inference(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():

            outputs = self.forward(x)
            probabilities = F.sigmoid(outputs)

            # Create a tensor with True where probabilities exceed the threshold
            predicted_classes = (probabilities > threshold).int()

            predicted_classes = predicted_classes.cpu().numpy()

        return predicted_classes