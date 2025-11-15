"""
RNN-based (LSTM/GRU) deepfake detector for audio.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This model is designed for DEFENSIVE purposes: detecting audio deepfakes.
"""

import torch
import torch.nn as nn


class RNNDetector(nn.Module):
    """
    RNN-based audio deepfake detector using LSTM/GRU.
    Operates on temporal features (spectrograms, MFCCs).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize RNN detector.

        Args:
            input_dim: Input feature dimension (e.g., n_mels)
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            rnn_type: 'lstm' or 'gru'
            bidirectional: Use bidirectional RNN
            dropout: Dropout rate
        """
        super(RNNDetector, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # RNN layer
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")

        # Output dimension
        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, freq, time)

        Returns:
            Output logits (batch, num_classes)
        """
        # Transpose to (batch, time, freq) for RNN
        x = x.transpose(1, 2)

        # RNN forward
        rnn_out, _ = self.rnn(x)

        # Use last output
        if self.bidirectional:
            # Concatenate forward and backward last outputs
            forward_out = rnn_out[:, -1, :self.hidden_dim]
            backward_out = rnn_out[:, 0, self.hidden_dim:]
            out = torch.cat([forward_out, backward_out], dim=1)
        else:
            out = rnn_out[:, -1, :]

        # Classify
        logits = self.classifier(out)

        return logits
