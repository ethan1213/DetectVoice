"""
Projected Gradient Descent (PGD) attack for adversarial robustness testing.

Reference: Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)

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


class PGD:
    """
    Projected Gradient Descent (PGD) attack.

    Iterative version of FGSM with projection to epsilon ball.
    Stronger than FGSM and often used for adversarial training.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 10,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        targeted: bool = False,
        random_start: bool = True
    ):
        """
        Initialize PGD attack.

        Args:
            model: Target model to attack
            epsilon: Maximum perturbation (L-infinity norm)
            alpha: Step size per iteration
            num_iter: Number of iterations
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping
            targeted: Whether to perform targeted attack
            random_start: Start from random point in epsilon ball
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted
        self.random_start = random_start

        self.model.eval()

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples using PGD.

        Args:
            x: Input tensor (batch, ...)
            y: True labels
            loss_fn: Loss function (default: CrossEntropyLoss)

        Returns:
            Tuple of (adversarial examples, metrics dict)
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        x = x.detach().clone()

        # Random initialization
        if self.random_start:
            x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
            x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)
            if self.clip_min is not None and self.clip_max is not None:
                x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)
        else:
            x_adv = x.clone()

        # PGD iterations
        for i in range(self.num_iter):
            x_adv.requires_grad = True

            # Forward pass
            outputs = self.model(x_adv)

            # Compute loss
            if self.targeted:
                loss = -loss_fn(outputs, y)
            else:
                loss = loss_fn(outputs, y)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

            # Get gradient
            grad = x_adv.grad.detach()

            # Update adversarial example
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project back to epsilon ball
            perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + perturbation

            # Clip to valid range
            if self.clip_min is not None and self.clip_max is not None:
                x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

        # Compute metrics
        perturbation = (x_adv - x).detach()
        l2_norm = torch.norm(perturbation.view(x.size(0), -1), p=2, dim=1).mean().item()
        linf_norm = torch.norm(perturbation.view(x.size(0), -1), p=float('inf'), dim=1).mean().item()

        # Compute SNR
        try:
            snr = compute_snr(x.detach(), x_adv.detach())
        except:
            snr = None

        metrics = {
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'snr_db': snr,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'num_iter': self.num_iter
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


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_iter: int = 10,
    loss_fn: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Simple PGD attack function.

    Args:
        model: Target model
        x: Input tensor
        y: Labels
        epsilon: Maximum perturbation
        alpha: Step size
        num_iter: Number of iterations
        loss_fn: Loss function

    Returns:
        Adversarial examples
    """
    attack = PGD(model=model, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
    x_adv, _ = attack.generate(x, y, loss_fn)
    return x_adv
