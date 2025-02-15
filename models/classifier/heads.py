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
        self.activation = nn.ReLU()

        self.layer2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout2 = nn.Dropout(dropout)

        self.layer3 = nn.Linear(hidden_size//2, hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        self.dropout3 = nn.Dropout(dropout)

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

        #x = self.layer3(x)
        #x = self.bn3(x)
        #x = self.activation(x)
        #x = self.dropout3(x)

        x = self.linear4(x)
        return x


class VanillaTokenPositionalEncoding(nn.Module):
    """
    Standard Transformer-style positional encoding for an instance of shape (5, 1280).
    Uses sinusoidal encodings to represent token positions.
    """

    def __init__(self, d_model=1280, seq_len=5):
        """
        Args:
            d_model (int): Embedding dimension size (default=1280).
            seq_len (int): Number of sequence positions (default=5 for chunks).
        """
        super(VanillaTokenPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Initialize the positional encoding matrix
        pe = torch.zeros(seq_len, d_model)

        # Compute positions: [0, 1, 2, 3, 4] for 5 chunks
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # Shape: (5, 1)

        # Compute the sinusoidal frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices and cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Sin for even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Cos for odd dimensions

        # Register buffer to prevent updating during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len=5, d_model=1280).

        Returns:
            x: Tensor with positional encodings added.
        """
        return x + self.pe[:x.size(1)]


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int = 1280, max_len: int = 5):
        super(RotaryPositionalEmbedding, self).__init__()
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
    def __init__(self, input_size=1280, hidden_size=256, output_size=2, dropout=0.5):
        super(ClsAttn, self).__init__()
        self.rotary_encoding = RotaryPositionalEmbedding(input_size)  # Corrected
        self.chunked_encoding = VanillaTokenPositionalEncoding(input_size)  # Corrected
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