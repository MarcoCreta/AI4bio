import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import torch.nn.functional as F
from config import Config
import torch.nn as nn
import math


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

        #x = self.layer3(x)
        #x = self.bn3(x)
        #x = self.activation(x)
        #x = self.dropout3(x)

        x = self.linear4(x)  # No activation here (CrossEntropyLoss applies softmax)
        return x

class ChunkPositionalEncoding(nn.Module):
    """
    Positional Encoding for Chunked Sequences with Overlapping Windows.
    Each chunk gets **two positional encodings**:
    - `pe_start`: Encoding for chunk start positions.
    - `pe_end`: Encoding for chunk end positions (considering overlap).
    """

    def __init__(self, d_model=1280, max_chunks=5, chunk_stride=256, chunk_size=512):
        super(ChunkPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_chunks = max_chunks
        self.chunk_stride = chunk_stride  # Shift of 256 tokens
        self.chunk_size = chunk_size  # 512-token chunks

        # Compute start positions
        chunk_starts = torch.arange(0, max_chunks * chunk_stride, chunk_stride, dtype=torch.float).unsqueeze(1)
        chunk_ends = chunk_starts + chunk_size  # Compute end positions

        # Sinusoidal encoding formula
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Compute start position encoding
        pe_start = torch.zeros(max_chunks, d_model)
        pe_start[:, 0::2] = torch.sin(chunk_starts * div_term)  # Sin for even indices
        pe_start[:, 1::2] = torch.cos(chunk_starts * div_term)  # Cos for odd indices

        # Compute end position encoding (same logic but for `chunk_ends`)
        pe_end = torch.zeros(max_chunks, d_model)
        pe_end[:, 0::2] = torch.sin(chunk_ends * div_term)
        pe_end[:, 1::2] = torch.cos(chunk_ends * div_term)

        # Register buffers (store as non-trainable)
        self.register_buffer('pe_start', pe_start)
        self.register_buffer('pe_end', pe_end)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_chunks=5, d_model=1280)

        Returns:
            x with start and end positional encodings **smoothed together**.
        """
        batch_size, num_chunks, d_model = x.shape

        # Compute smooth transition between start and end positional encodings
        alpha = torch.linspace(0, 1, num_chunks, device=x.device).view(1, num_chunks, 1)  # Interpolation weight
        pe_combined = (1 - alpha) * self.pe_start[:num_chunks] + alpha * self.pe_end[:num_chunks]

        x = x + pe_combined  # Apply smoothed encoding
        return x  # (batch_size, num_chunks=5, d_model=1280)

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

class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model=1280, seq_len=512, num_chunks=5, eps=1e-6):
        """
        Hybrid Positional Encoding:
        - Computes sinusoidal encoding for 512 tokens (shifted by 256)
        - Projects it to 1280 dimensions safely
        - Applies chunk-level RoPE with stabilization
        """
        super(HybridPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_chunks = num_chunks
        self.eps = eps  # Small value to prevent NaN issues

        # Standard sinusoidal positional encoding for 512 tokens
        pe = torch.zeros(seq_len, seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (512, 1)
        div_term = torch.exp(torch.arange(0, seq_len, 2).float() * (-math.log(10000.0) / seq_len))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # Store as a buffer (not trainable)

        # Projection layer to map 512-D positional encoding to 1280-D (with weight initialization)
        self.projection = nn.Linear(seq_len, d_model)
        nn.init.xavier_uniform_(self.projection.weight)  # Better weight initialization

        # LayerNorm for stability
        self.norm = nn.LayerNorm(d_model)

        # Scaling factor (learned, initialized close to 1)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        # RoPE for chunk-wise encoding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_chunks=5, d_model=1280)

        Returns:
            x with hybrid positional encoding applied.
        """
        batch_size, num_chunks, d_model = x.shape
        device = x.device

        # 1️⃣ Get positional encodings for original 512-token sequences
        pe_expanded = self.pe.unsqueeze(0).expand(batch_size, -1, -1).to(device)  # (B, 512, 512)

        # 2️⃣ Safe Projection to 1280-D
        pe_projected = self.projection(pe_expanded)  # (B, 512, 1280)
        pe_projected = self.norm(pe_projected)  # Apply LayerNorm for stability
        pe_projected = self.scale * pe_projected  # Scale down to avoid large values

        # 3️⃣ Assign Positional Encoding to Chunks (Avoiding NaNs)
        pe_chunks = torch.zeros((batch_size, num_chunks, d_model), device=device)

        for i in range(num_chunks):
            start_idx = i * 256  # Shift by 256 for each chunk
            pe_chunks[:, i, :] = pe_projected[:, start_idx:start_idx+512, :].mean(dim=1)  # More stable pooling

        # 4️⃣ Apply Chunk-Level RoPE
        chunk_positions = torch.arange(num_chunks, dtype=torch.float32, device=device).unsqueeze(1)  # (5, 1)
        chunk_freqs = torch.outer(chunk_positions.squeeze(), self.inv_freq)  # (5, dim // 2)

        sin_c, cos_c = chunk_freqs.sin(), chunk_freqs.cos()
        sin_c, cos_c = sin_c.unsqueeze(0), cos_c.unsqueeze(0)  # Expand for batch size

        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        x_rotated = torch.cat([x_even * cos_c - x_odd * sin_c, x_even * sin_c + x_odd * cos_c], dim=-1)

        # 5️⃣ Final Encoding: Combine token-based and chunk-based positional info
        x_final = x_rotated + pe_chunks  # Inject both token-aware and chunk-aware positional info

        return x_final  # (B, 5, 1280)


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