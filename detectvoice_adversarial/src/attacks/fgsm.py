"""
Fast Gradient Sign Method (FGSM) attack for adversarial robustness testing.

Reference: Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (2014)

⚠️  SECURITY & ETHICS NOTICE ⚠️
This implementation is for DEFENSIVE research purposes only:
- Testing model robustness
- Adversarial training
- Security evaluation
DO NOT use for malicious purposes or unauthorized system attacks.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np
from src.utils.audio import compute_snr
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FGSM:
    """
    Fast Gradient Sign Method (FGSM) attack.

    Generates adversarial examples by adding small perturbations
    in the direction of the gradient sign.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        targeted: bool = False
    ):
        """
        Initialize FGSM attack.

        Args:
            model: Target model to attack
            epsilon: Perturbation magnitude
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping
            targeted: Whether to perform targeted attack
        """
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

        self.model.eval()

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples using FGSM.

        Args:
            x: Input tensor (batch, ...)
            y: True labels
            loss_fn: Loss function (default: CrossEntropyLoss)

        Returns:
            Tuple of (adversarial examples, metrics dict)
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        # Ensure input requires gradient
        x = x.detach().clone()
        x.requires_grad = True

        # Forward pass
        outputs = self.model(x)

        # Compute loss
        if self.targeted:
            loss = -loss_fn(outputs, y)
        else:
            loss = loss_fn(outputs, y)

        # Backward pass
        self.model.zero_grad()
        loss.backward()

        # Get gradient sign
        grad_sign = x.grad.sign()

        # Generate adversarial example
        x_adv = x + self.epsilon * grad_sign

        # Clip to valid range
        if self.clip_min is not None and self.clip_max is not None:
            x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        # Compute metrics
        perturbation = (x_adv - x).detach()
        l2_norm = torch.norm(perturbation.view(x.size(0), -1), p=2, dim=1).mean().item()
        linf_norm = torch.norm(perturbation.view(x.size(0), -1), p=float('inf'), dim=1).mean().item()

        # Compute SNR if possible
        try:
            snr = compute_snr(x.detach(), x_adv.detach())
        except:
            snr = None

        metrics = {
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'snr_db': snr,
            'epsilon': self.epsilon
        }

        return x_adv.detach(), metrics

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> dict:
        """
        Evaluate attack success rate.

        Args:
            x: Input tensor
            y: True labels
            loss_fn: Loss function

        Returns:
            Dictionary with evaluation metrics
        """
        # Generate adversarial examples
        x_adv, attack_metrics = self.generate(x, y, loss_fn)

        # Evaluate on clean samples
        with torch.no_grad():
            clean_outputs = self.model(x)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_acc = (clean_preds == y).float().mean().item()

        # Evaluate on adversarial samples
        with torch.no_grad():
            adv_outputs = self.model(x_adv)
            adv_preds = adv_outputs.argmax(dim=1)
            adv_acc = (adv_preds == y).float().mean().item()

        # Attack success rate
        attack_success_rate = (clean_preds != adv_preds).float().mean().item()

        eval_metrics = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'attack_success_rate': attack_success_rate,
            'robustness_drop': clean_acc - adv_acc,
            **attack_metrics
        }

        return eval_metrics


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    loss_fn: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Simple FGSM attack function.

    Args:
        model: Target model
        x: Input tensor
        y: Labels
        epsilon: Perturbation magnitude
        loss_fn: Loss function

    Returns:
        Adversarial examples
    """
    attack = FGSM(model=model, epsilon=epsilon)
    x_adv, _ = attack.generate(x, y, loss_fn)
    return x_adv
