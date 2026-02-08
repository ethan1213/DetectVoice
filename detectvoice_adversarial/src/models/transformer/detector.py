"""
Transformer-based deepfake detector for audio.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This model is designed for DEFENSIVE purposes: detecting audio deepfakes.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerDetector(nn.Module):
    """
    Transformer-based audio deepfake detector.
    Operates on sequential features (spectrogram frames).
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Transformer detector.

        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(TransformerDetector, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, freq, time)

        Returns:
            Output logits (batch, num_classes)
        """
        # Transpose to (batch, time, freq)
        x = x.transpose(1, 2)

        batch_size, seq_len, _ = x.size()

        # Project to d_model
        x = self.input_projection(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Use CLS token output
        cls_output = x[:, 0, :]

        # Classify
        logits = self.classifier(cls_output)

        return logits
