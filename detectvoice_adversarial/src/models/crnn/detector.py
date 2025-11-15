"""
CRNN (CNN + RNN) deepfake detector for audio.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This model is designed for DEFENSIVE purposes: detecting audio deepfakes.
"""

import torch
import torch.nn as nn


class CRNNDetector(nn.Module):
    """
    CRNN detector combining CNN for feature extraction and RNN for temporal modeling.
    """

    def __init__(
        self,
        input_channels: int = 1,
        cnn_channels: list = [64, 128, 256],
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize CRNN detector.

        Args:
            input_channels: Number of input channels
            cnn_channels: List of CNN channel dimensions
            rnn_hidden_dim: RNN hidden dimension
            rnn_num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(CRNNDetector, self).__init__()

        # CNN layers
        cnn_layers = []
        in_ch = input_channels

        for out_ch in cnn_channels:
            cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2, 1))  # Pool only in frequency
                )
            )
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # RNN layer (bidirectional LSTM)
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_num_layers > 1 else 0
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, freq, time) or (batch, 1, freq, time)

        Returns:
            Output logits (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN feature extraction
        x = self.cnn(x)  # (batch, channels, freq', time)

        # Reshape for RNN: (batch, time, channels * freq')
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, time, channels * freq)

        # RNN temporal modeling
        rnn_out, _ = self.rnn(x)  # (batch, time, hidden*2)

        # Use last output
        out = rnn_out[:, -1, :]

        # Classify
        logits = self.classifier(out)

        return logits
