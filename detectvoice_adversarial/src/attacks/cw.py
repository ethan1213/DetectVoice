"""
Carlini & Wagner (C&W) L2 and L-infinity attacks.

Reference: Carlini & Wagner "Towards Evaluating the Robustness of Neural Networks" (2017)

⚠️  SECURITY & ETHICS NOTICE ⚠️
This implementation is for DEFENSIVE research purposes only:
- Testing model robustness
- Adversarial training
- Security evaluation
DO NOT use for malicious purposes or unauthorized system attacks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import numpy as np
from src.utils.audio import compute_snr
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CarliniWagnerL2:
    """
    Carlini & Wagner L2 attack.

    Optimizes perturbation to minimize L2 distance while causing misclassification.
    Generally more powerful but slower than FGSM/PGD.
    """

    def __init__(
        self,
        model: nn.Module,
        c: float = 1.0,
        kappa: float = 0.0,
        max_iter: int = 100,
        learning_rate: float = 0.01,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        targeted: bool = False
    ):
        """
        Initialize C&W L2 attack.

        Args:
            model: Target model to attack
            c: Confidence parameter (trade-off between loss and perturbation)
            kappa: Confidence margin
            max_iter: Maximum optimization iterations
            learning_rate: Learning rate for optimizer
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping
            targeted: Whether to perform targeted attack
        """
        self.model = model
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.targeted = targeted

        self.model.eval()

    def _f(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the objective function for C&W attack.

        Args:
            outputs: Model outputs (logits)
            y: True labels

        Returns:
            Objective value
        """
        # Get top-2 logits
        sorted_logits, _ = torch.sort(outputs, dim=1, descending=True)

        # True class logit
        true_logit = outputs.gather(1, y.unsqueeze(1)).squeeze(1)

        # Best other class logit
        mask = torch.zeros_like(outputs).scatter_(1, y.unsqueeze(1), 1.0).bool()
        other_logit = outputs.masked_fill(mask, float('-inf')).max(dim=1)[0]

        if self.targeted:
            # Targeted: maximize target class
            loss = torch.clamp(other_logit - true_logit + self.kappa, min=0.0)
        else:
            # Untargeted: minimize true class
            loss = torch.clamp(true_logit - other_logit + self.kappa, min=0.0)

        return loss

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples using C&W L2.

        Args:
            x: Input tensor (batch, ...)
            y: True labels

        Returns:
            Tuple of (adversarial examples, metrics dict)
        """
        batch_size = x.size(0)
        x = x.detach().clone()

        # Initialize perturbation variable (in tanh space for box constraints)
        if self.clip_min is not None and self.clip_max is not None:
            # Map x to tanh space
            x_tanh = self._to_tanh_space(x)
            w = torch.zeros_like(x_tanh, requires_grad=True)
        else:
            w = torch.zeros_like(x, requires_grad=True)

        # Optimizer
        optimizer = optim.Adam([w], lr=self.learning_rate)

        best_adv = x.clone()
        best_l2 = float('inf') * torch.ones(batch_size, device=x.device)

        for iteration in range(self.max_iter):
            optimizer.zero_grad()

            # Get adversarial example
            if self.clip_min is not None and self.clip_max is not None:
                x_adv = self._from_tanh_space(x_tanh + w)
            else:
                x_adv = x + w

            # Forward pass
            outputs = self.model(x_adv)

            # Compute loss
            f_loss = self._f(outputs, y)
            l2_dist = torch.norm((x_adv - x).view(batch_size, -1), p=2, dim=1)

            loss = l2_dist + self.c * f_loss.sum()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update best adversarial examples
            preds = outputs.argmax(dim=1)
            successful = (preds != y) if not self.targeted else (preds == y)

            for i in range(batch_size):
                if successful[i] and l2_dist[i] < best_l2[i]:
                    best_l2[i] = l2_dist[i]
                    best_adv[i] = x_adv[i].detach()

        # Compute metrics
        perturbation = (best_adv - x).detach()
        l2_norm = torch.norm(perturbation.view(batch_size, -1), p=2, dim=1).mean().item()
        linf_norm = torch.norm(perturbation.view(batch_size, -1), p=float('inf'), dim=1).mean().item()

        try:
            snr = compute_snr(x.detach(), best_adv.detach())
        except:
            snr = None

        metrics = {
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'snr_db': snr,
            'c': self.c,
            'max_iter': self.max_iter
        }

        return best_adv, metrics

    def _to_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """Map from [clip_min, clip_max] to tanh space."""
        x_normalized = (x - self.clip_min) / (self.clip_max - self.clip_min)
        x_normalized = torch.clamp(x_normalized, 0.0, 1.0)
        return torch.atanh((x_normalized - 0.5) * 2 * 0.999999)

    def _from_tanh_space(self, x_tanh: torch.Tensor) -> torch.Tensor:
        """Map from tanh space to [clip_min, clip_max]."""
        x_normalized = (torch.tanh(x_tanh) / 2 + 0.5)
        return x_normalized * (self.clip_max - self.clip_min) + self.clip_min

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> dict:
        """Evaluate attack success rate."""
        x_adv, attack_metrics = self.generate(x, y)

        with torch.no_grad():
            clean_outputs = self.model(x)
            clean_preds = clean_outputs.argmax(dim=1)
            clean_acc = (clean_preds == y).float().mean().item()

            adv_outputs = self.model(x_adv)
            adv_preds = adv_outputs.argmax(dim=1)
            adv_acc = (adv_preds == y).float().mean().item()

        attack_success_rate = (clean_preds != adv_preds).float().mean().item()

        eval_metrics = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'attack_success_rate': attack_success_rate,
            'robustness_drop': clean_acc - adv_acc,
            **attack_metrics
        }

        return eval_metrics


def cw_l2_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 1.0,
    max_iter: int = 100
) -> torch.Tensor:
    """
    Simple C&W L2 attack function.

    Args:
        model: Target model
        x: Input tensor
        y: Labels
        c: Confidence parameter
        max_iter: Maximum iterations

    Returns:
        Adversarial examples
    """
    attack = CarliniWagnerL2(model=model, c=c, max_iter=max_iter)
    x_adv, _ = attack.generate(x, y)
    return x_adv
