"""
CNN-based deepfake detector for audio.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This model is designed for DEFENSIVE purposes: detecting audio deepfakes.
Use responsibly and in compliance with laws and ethical guidelines.
"""

import torch
import torch.nn as nn


class CNNDetector(nn.Module):
    """
    CNN-based audio deepfake detector.
    Operates on mel-spectrograms.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        """
        Initialize CNN detector.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate
        """
        super(CNNDetector, self).__init__()

        # Convolutional blocks
        self.conv1 = self._make_conv_block(input_channels, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, channels, freq, time) or (batch, freq, time)

        Returns:
            Output logits (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings before classification."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x
