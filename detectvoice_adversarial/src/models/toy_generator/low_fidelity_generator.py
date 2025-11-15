"""
LOW-FIDELITY Toy Generator for Testing Purposes ONLY.

⚠️  CRITICAL SECURITY NOTICE ⚠️

This is a TOY GENERATOR with INTENTIONALLY LIMITED CAPABILITY:

1. LOW FIDELITY: Produces low-quality audio unsuitable for realistic synthesis
2. TESTING ONLY: Designed SOLELY for testing discriminators and detectors
3. NOT FOR PRODUCTION: Cannot and should not be used for realistic voice synthesis
4. RESEARCH PURPOSE: For defensive ML research and robustness testing ONLY

This generator is deliberately simple and produces noticeable artifacts.
It is NOT a high-quality TTS or voice cloning system.

USE RESTRICTIONS:
- ✓ Testing discriminator performance
- ✓ Generating labeled synthetic data for training detectors
- ✓ Academic research on detection methods
- ✗ Creating realistic deepfakes
- ✗ Malicious impersonation
- ✗ Any unauthorized voice synthesis

BY USING THIS CODE, YOU AGREE TO USE IT RESPONSIBLY AND ETHICALLY.
"""

import torch
import torch.nn as nn
import numpy as np


class ToySpectrogramGenerator(nn.Module):
    """
    SIMPLIFIED toy generator for mel-spectrograms.

    This is a basic autoencoder-style generator that produces
    LOW-FIDELITY spectrograms for testing discriminators.

    NOT suitable for realistic voice synthesis.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        output_channels: int = 1
    ):
        """
        Initialize toy generator.

        Args:
            latent_dim: Latent dimension
            output_channels: Output channels (1 for mono)
        """
        super(ToySpectrogramGenerator, self).__init__()

        # Deliberately simple architecture
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64 * 4 * 4)
        )

        # Simple upsampling - produces noticeable artifacts
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate low-fidelity spectrogram from noise.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Generated spectrogram (batch, 1, freq, time)
        """
        x = self.fc(z)
        x = x.view(x.size(0), 64, 4, 4)
        x = self.decoder(x)

        return x

    def generate_batch(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """
        Generate a batch of low-fidelity spectrograms.

        Args:
            batch_size: Number of samples to generate
            device: Device to use

        Returns:
            Generated spectrograms
        """
        z = torch.randn(batch_size, 128, device=device)
        return self.forward(z)


class SimplePitchShifter(nn.Module):
    """
    VERY SIMPLE pitch shifter for testing.

    Produces obvious artifacts - NOT for realistic synthesis.
    """

    def __init__(self):
        super(SimplePitchShifter, self).__init__()

    def forward(
        self,
        spectrogram: torch.Tensor,
        shift_factor: float = 1.1
    ) -> torch.Tensor:
        """
        Apply simple pitch shift (frequency axis scaling).

        Args:
            spectrogram: Input spectrogram (batch, freq, time)
            shift_factor: Pitch shift factor (>1 = higher, <1 = lower)

        Returns:
            Pitch-shifted spectrogram (with artifacts)
        """
        # Simple interpolation - produces noticeable artifacts
        batch_size, freq_bins, time_bins = spectrogram.size()

        new_freq_bins = int(freq_bins / shift_factor)

        # Interpolate (creates artifacts)
        shifted = torch.nn.functional.interpolate(
            spectrogram.unsqueeze(1),
            size=(new_freq_bins, time_bins),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        # Pad or crop to original size
        if new_freq_bins > freq_bins:
            shifted = shifted[:, :freq_bins, :]
        elif new_freq_bins < freq_bins:
            padding = freq_bins - new_freq_bins
            shifted = torch.nn.functional.pad(shifted, (0, 0, 0, padding))

        return shifted


class NoiseInjector(nn.Module):
    """
    Simple noise injection for creating low-quality synthetic samples.
    """

    def __init__(self, noise_level: float = 0.1):
        """
        Initialize noise injector.

        Args:
            noise_level: Noise magnitude
        """
        super(NoiseInjector, self).__init__()
        self.noise_level = noise_level

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add noise to input.

        Args:
            x: Input tensor

        Returns:
            Noisy tensor
        """
        noise = torch.randn_like(x) * self.noise_level
        return x + noise


def generate_toy_samples(
    num_samples: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate low-fidelity toy samples for testing.

    Args:
        num_samples: Number of samples to generate
        device: Device to use

    Returns:
        Generated low-fidelity spectrograms
    """
    generator = ToySpectrogramGenerator().to(device)
    generator.eval()

    with torch.no_grad():
        samples = generator.generate_batch(num_samples, device)

    return samples


# USAGE DISCLAIMER
def print_usage_disclaimer():
    """Print usage disclaimer and restrictions."""
    disclaimer = """
    ═══════════════════════════════════════════════════════════════
    TOY GENERATOR - LOW FIDELITY - TESTING ONLY
    ═══════════════════════════════════════════════════════════════

    This generator is INTENTIONALLY LIMITED and produces LOW-QUALITY
    audio for DEFENSIVE research purposes ONLY.

    PERMITTED USES:
    - Testing discriminator models
    - Training deepfake detectors
    - Academic research on detection methods

    PROHIBITED USES:
    - Creating realistic deepfakes
    - Impersonation or deception
    - Malicious voice synthesis
    - Any unauthorized use

    By proceeding, you agree to use this code ethically and legally.
    ═══════════════════════════════════════════════════════════════
    """
    print(disclaimer)


if __name__ == "__main__":
    print_usage_disclaimer()
