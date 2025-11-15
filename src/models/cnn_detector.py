"""
CNN-based detector for audio deepfake detection.
Includes freeze, export, and evaluation capabilities.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from src.utils.model_export import (
    freeze_model,
    save_pytorch,
    save_torchscript,
    save_onnx,
    export_all_formats
)


class CNNDetector(nn.Module):
    """
    CNN-based audio deepfake detector.
    Compatible with mel-spectrograms.
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

        self.conv1 = self._make_conv_block(input_channels, 64)
        self.conv2 = self._make_conv_block(64, 128)
        self.conv3 = self._make_conv_block(128, 256)
        self.conv4 = self._make_conv_block(256, 512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def _make_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create convolutional block."""
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
        """
        Extract feature embeddings before classification.

        Args:
            x: Input tensor

        Returns:
            Feature embeddings (batch, 512)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        return x

    def freeze(self):
        """Freeze all model parameters."""
        return freeze_model(self)

    def save(
        self,
        save_path: Path,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[dict] = None
    ):
        """
        Save model as PyTorch checkpoint.

        Args:
            save_path: Path to save checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
        """
        save_pytorch(self, save_path, optimizer, epoch, metrics)

    def export_torchscript(
        self,
        save_path: Path,
        example_input: torch.Tensor
    ):
        """
        Export model to TorchScript.

        Args:
            save_path: Path to save TorchScript model
            example_input: Example input for tracing
        """
        save_torchscript(self, save_path, example_input)

    def export_onnx(
        self,
        save_path: Path,
        example_input: torch.Tensor
    ):
        """
        Export model to ONNX.

        Args:
            save_path: Path to save ONNX model
            example_input: Example input for export
        """
        save_onnx(self, save_path, example_input)

    def export_all(
        self,
        base_path: Path,
        model_name: str,
        example_input: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[dict] = None
    ):
        """
        Export model to all formats.

        Args:
            base_path: Base directory for saving
            model_name: Name of the model
            example_input: Example input tensor
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)

        Returns:
            Dictionary with paths to saved models
        """
        return export_all_formats(
            self, base_path, model_name, example_input,
            optimizer, epoch, metrics
        )

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cpu'
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: Data loader
            device: Device to use

        Returns:
            Tuple of (accuracy, y_true, y_pred, y_scores)
        """
        self.eval()
        self.to(device)

        all_preds = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_scores = np.array(all_scores)

        accuracy = (y_true == y_pred).mean()

        return accuracy, y_true, y_pred, y_scores
