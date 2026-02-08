"""
GAN Discriminators for audio forensics.

These discriminators are designed for DEFENSIVE purposes:
- Detecting GAN-generated audio
- Audio forensics analysis
- Training robust detectors

⚠️  SECURITY & ETHICS NOTICE ⚠️
These models are for DEFENSIVE research and forensics.
NOT for creating undetectable deepfakes.
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN-style discriminator for audio spectrograms.
    Classifies patches as real/fake for finer-grained detection.
    """

    def __init__(
        self,
        input_channels: int = 1,
        ndf: int = 64,
        n_layers: int = 3
    ):
        """
        Initialize PatchGAN discriminator.

        Args:
            input_channels: Number of input channels
            ndf: Base number of filters
            n_layers: Number of layers
        """
        super(PatchGANDiscriminator, self).__init__()

        layers = []

        # First layer
        layers.append(
            nn.Sequential(
                nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(ndf * nf_mult),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Final layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(
            nn.Sequential(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )

        # Output layer
        layers.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram (batch, channels, freq, time)

        Returns:
            Patch predictions (batch, 1, H, W)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for audio.
    Operates at different resolutions to capture features at multiple scales.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_scales: int = 3,
        ndf: int = 64
    ):
        """
        Initialize multi-scale discriminator.

        Args:
            input_channels: Number of input channels
            num_scales: Number of scales
            ndf: Base number of filters
        """
        super(MultiScaleDiscriminator, self).__init__()

        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        for i in range(num_scales):
            self.discriminators.append(
                PatchGANDiscriminator(input_channels, ndf)
            )

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass through all scales.

        Args:
            x: Input spectrogram

        Returns:
            List of outputs from each scale
        """
        outputs = []
        for i, discriminator in enumerate(self.discriminators):
            outputs.append(discriminator(x))
            if i < self.num_scales - 1:
                x = self.downsample(x)

        return outputs


class SelfAttentionDiscriminator(nn.Module):
    """
    Discriminator with self-attention for capturing long-range dependencies.
    Useful for detecting subtle artifacts in generated audio.
    """

    def __init__(
        self,
        input_channels: int = 1,
        ndf: int = 64
    ):
        """
        Initialize self-attention discriminator.

        Args:
            input_channels: Number of input channels
            ndf: Base number of filters
        """
        super(SelfAttentionDiscriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.self_attention = SelfAttention(ndf * 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.self_attention(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)

        return x


class SelfAttention(nn.Module):
    """Self-attention module."""

    def __init__(self, in_channels: int):
        super(SelfAttention, self).__init__()

        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size, C, H, W = x.size()

        # Query, Key, Value projections
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, H*W, C')
        key = self.key(x).view(batch_size, -1, H * W)  # (B, C', H*W)
        value = self.value(x).view(batch_size, -1, H * W)  # (B, C, H*W)

        # Attention
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = torch.softmax(attention, dim=-1)

        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, C, H, W)

        # Residual connection
        out = self.gamma * out + x

        return out


class ForensicFeatureExtractor(nn.Module):
    """
    Feature extractor for forensic analysis.
    Extracts intermediate features useful for detecting artifacts.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2
    ):
        """
        Initialize forensic feature extractor.

        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
        """
        super(ForensicFeatureExtractor, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract forensic features."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = self.features(x)
        features = features.view(features.size(0), -1)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.extract_features(x)
        logits = self.classifier(features)

        return logits
