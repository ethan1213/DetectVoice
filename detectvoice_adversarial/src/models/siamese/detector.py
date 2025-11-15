"""
Siamese Network for audio deepfake detection and speaker verification.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This model is designed for DEFENSIVE purposes: detecting audio deepfakes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseDetector(nn.Module):
    """
    Siamese Network for comparing audio pairs.
    Can detect voice cloning by comparing embeddings.
    """

    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 128
    ):
        """
        Initialize Siamese detector.

        Args:
            input_channels: Number of input channels
            embedding_dim: Embedding dimension
        """
        super(SiameseDetector, self).__init__()

        # Shared embedding network
        self.embedding_net = nn.Sequential(
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

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one input.

        Args:
            x: Input tensor (batch, freq, time) or (batch, 1, freq, time)

        Returns:
            Embedding (batch, embedding_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.embedding_net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)

        return x

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor
    ) -> tuple:
        """
        Forward pass for a pair of inputs.

        Args:
            x1: First input
            x2: Second input

        Returns:
            Tuple of (embedding1, embedding2)
        """
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)

        return emb1, emb2

    def compute_distance(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        distance_type: str = 'euclidean'
    ) -> torch.Tensor:
        """
        Compute distance between embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding
            distance_type: 'euclidean' or 'cosine'

        Returns:
            Distance tensor
        """
        if distance_type == 'euclidean':
            return torch.norm(emb1 - emb2, p=2, dim=1)
        elif distance_type == 'cosine':
            return 1 - F.cosine_similarity(emb1, emb2, dim=1)
        else:
            raise ValueError(f"Unknown distance type: {distance_type}")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.

        Args:
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            emb1: First embedding
            emb2: Second embedding
            label: 1 for similar, 0 for dissimilar

        Returns:
            Loss value
        """
        distance = torch.norm(emb1 - emb2, p=2, dim=1)

        loss_positive = label * distance.pow(2)
        loss_negative = (1 - label) * F.relu(self.margin - distance).pow(2)

        loss = 0.5 * (loss_positive + loss_negative)

        return loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet loss for Siamese networks.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embedding
            positive: Positive embedding (same class)
            negative: Negative embedding (different class)

        Returns:
            Loss value
        """
        pos_distance = torch.norm(anchor - positive, p=2, dim=1)
        neg_distance = torch.norm(anchor - negative, p=2, dim=1)

        loss = F.relu(pos_distance - neg_distance + self.margin)

        return loss.mean()
