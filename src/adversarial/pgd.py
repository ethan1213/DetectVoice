"""
PGD (Projected Gradient Descent) attack.
Optional adversarial module for DetectVoice.

⚠️ FOR DEFENSIVE RESEARCH AND TESTING ONLY
"""

import torch
import torch.nn as nn
from typing import Tuple


class PGD:
    """PGD attack (iterative FGSM)."""

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.03,
        alpha: float = 0.01,
        num_iter: int = 10,
        device: str = 'cpu'
    ):
        """
        Initialize PGD attack.

        Args:
            model: Target model
            epsilon: Maximum perturbation
            alpha: Step size per iteration
            num_iter: Number of iterations
            device: Device to use
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.device = device

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples using PGD.

        Args:
            x: Input tensor
            y: Labels

        Returns:
            Tuple of (adversarial examples, metrics)
        """
        self.model.eval()
        self.model.to(self.device)

        x = x.to(self.device)
        y = y.to(self.device)

        # Random initialization
        x_adv = x + torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, x - self.epsilon, x + self.epsilon)

        loss_fn = nn.CrossEntropyLoss()

        # PGD iterations
        for _ in range(self.num_iter):
            x_adv.requires_grad = True

            output = self.model(x_adv)
            loss = loss_fn(output, y)

            self.model.zero_grad()
            loss.backward()

            # Update
            grad = x_adv.grad.detach()
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project to epsilon ball
            perturbation = torch.clamp(x_adv - x, -self.epsilon, self.epsilon)
            x_adv = x + perturbation

        # Metrics
        perturbation = (x_adv - x).detach()
        l2_norm = torch.norm(perturbation.view(x.size(0), -1), p=2, dim=1).mean().item()
        linf_norm = torch.norm(perturbation.view(x.size(0), -1), p=float('inf'), dim=1).mean().item()

        metrics = {
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'epsilon': self.epsilon,
            'num_iter': self.num_iter
        }

        return x_adv.detach(), metrics

    def evaluate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> dict:
        """Evaluate attack success."""
        x_adv, attack_metrics = self.generate(x, y)

        with torch.no_grad():
            clean_out = self.model(x.to(self.device))
            clean_preds = clean_out.argmax(dim=1)
            clean_acc = (clean_preds == y.to(self.device)).float().mean().item()

            adv_out = self.model(x_adv)
            adv_preds = adv_out.argmax(dim=1)
            adv_acc = (adv_preds == y.to(self.device)).float().mean().item()

        return {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'robustness_drop': clean_acc - adv_acc,
            **attack_metrics
        }
