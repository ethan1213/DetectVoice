"""
Autoencoder-based deepfake detector for audio.

Uses reconstruction error as anomaly score for detection.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This model is designed for DEFENSIVE purposes: detecting audio deepfakes.
"""

import torch
import torch.nn as nn


class AudioAutoencoder(nn.Module):
    """
    Denoising Autoencoder for audio deepfake detection.
    Trained on real audio; high reconstruction error indicates fake.
    """

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 128
    ):
        """
        Initialize autoencoder.

        Args:
            input_channels: Number of input channels
            latent_dim: Latent space dimension
        """
        super(AudioAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Latent
        self.fc_encode = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        z = self.fc_encode(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute reconstruction error.

        Args:
            x: Input tensor
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Reconstruction error
        """
        x_recon = self.forward(x)

        if reduction == 'none':
            error = torch.mean((x - x_recon) ** 2, dim=[1, 2, 3])
        elif reduction == 'mean':
            error = torch.mean((x - x_recon) ** 2)
        elif reduction == 'sum':
            error = torch.sum((x - x_recon) ** 2)
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        return error


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for audio deepfake detection.
    """

    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 128
    ):
        """
        Initialize VAE.

        Args:
            input_channels: Number of input channels
            latent_dim: Latent space dimension
        """
        super(VariationalAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        # Latent (mu and logvar)
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> tuple:
        """Encode input to latent mu and logvar."""
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        x = self.fc_decode(z)
        x = x.view(x.size(0), 128, 4, 4)
        x = self.decoder(x)
        return x

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def vae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> tuple:
        """
        Compute VAE loss (reconstruction + KL divergence).

        Args:
            x: Original input
            x_recon: Reconstructed input
            mu: Latent mean
            logvar: Latent log variance
            beta: KL weight

        Returns:
            Tuple of (total_loss, recon_loss, kl_loss)
        """
        # Reconstruction loss
        recon_loss = torch.mean((x - x_recon) ** 2)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, recon_loss, kl_loss
