"""
FGSM (Fast Gradient Sign Method) attack.
Optional adversarial module for DetectVoice.

⚠️ FOR DEFENSIVE RESEARCH AND TESTING ONLY
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np


def fgsm_attack(
    model: nn.Module,
    waveform: torch.Tensor,
    label: torch.Tensor,
    epsilon: float = 0.03,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate adversarial example using FGSM.

    Args:
        model: Target model
        waveform: Input waveform or spectrogram
        label: True label
        epsilon: Perturbation magnitude
        device: Device to use

    Returns:
        Adversarial example
    """
    model.eval()
    model.to(device)

    waveform = waveform.to(device)
    label = label.to(device)

    # Ensure input requires gradient
    waveform_adv = waveform.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(waveform_adv)

    # Loss
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, label)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Generate adversarial example
    grad_sign = waveform_adv.grad.sign()
    waveform_adv = waveform_adv + epsilon * grad_sign

    return waveform_adv.detach()


class FGSM:
    """FGSM attack class with metrics."""

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        device: str = 'cpu'
    ):
        """
        Initialize FGSM attack.

        Args:
            model: Target model
            epsilon: Perturbation magnitude
            device: Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples with metrics.

        Args:
            x: Input tensor
            y: Labels

        Returns:
            Tuple of (adversarial examples, metrics)
        """
        x_adv = fgsm_attack(self.model, x, y, self.epsilon, self.device)

        # Compute metrics
        perturbation = (x_adv - x).detach()
        l2_norm = torch.norm(perturbation.view(x.size(0), -1), p=2, dim=1).mean().item()
        linf_norm = torch.norm(perturbation.view(x.size(0), -1), p=float('inf'), dim=1).mean().item()

        metrics = {
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'epsilon': self.epsilon
        }

        return x_adv, metrics

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> dict:
        """
        Evaluate attack success.

        Args:
            x: Input tensor
            y: Labels

        Returns:
            Evaluation metrics
        """
        # Generate adversarial examples
        x_adv, attack_metrics = self.generate(x, y)

        # Evaluate clean
        with torch.no_grad():
            clean_out = self.model(x.to(self.device))
            clean_preds = clean_out.argmax(dim=1)
            clean_acc = (clean_preds == y.to(self.device)).float().mean().item()

        # Evaluate adversarial
        with torch.no_grad():
            adv_out = self.model(x_adv)
            adv_preds = adv_out.argmax(dim=1)
            adv_acc = (adv_preds == y.to(self.device)).float().mean().item()

        metrics = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'robustness_drop': clean_acc - adv_acc,
            **attack_metrics
        }

        return metrics
