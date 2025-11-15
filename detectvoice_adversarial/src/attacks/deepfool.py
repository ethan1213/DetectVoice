"""
DeepFool attack for adversarial robustness testing.

Reference: Moosavi-Dezfooli et al. "DeepFool: a simple and accurate method to fool deep neural networks" (2016)

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


class DeepFool:
    """
    DeepFool attack.

    Finds minimal perturbation to cross decision boundary.
    Iteratively linearizes the classifier and finds closest boundary.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 2,
        max_iter: int = 50,
        overshoot: float = 0.02,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ):
        """
        Initialize DeepFool attack.

        Args:
            model: Target model to attack
            num_classes: Number of output classes
            max_iter: Maximum iterations
            overshoot: Overshoot parameter for numerical stability
            clip_min: Minimum value for clipping
            clip_max: Maximum value for clipping
        """
        self.model = model
        self.num_classes = num_classes
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.model.eval()

    def generate_single(
        self,
        x: torch.Tensor,
        y: int
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial example for a single input.

        Args:
            x: Input tensor (single sample)
            y: True label

        Returns:
            Tuple of (adversarial example, metrics)
        """
        x = x.clone().detach().unsqueeze(0)
        x.requires_grad = True

        # Get initial prediction
        with torch.no_grad():
            f = self.model(x)
            pred_label = f.argmax(dim=1).item()

        if pred_label != y:
            # Already misclassified
            return x.squeeze(0).detach(), {'iterations': 0, 'l2_norm': 0.0}

        x_adv = x.clone()
        iteration = 0

        while pred_label == y and iteration < self.max_iter:
            x_adv.requires_grad = True

            # Forward pass
            f = self.model(x_adv)

            # Get gradients for all classes
            gradients = []
            for k in range(self.num_classes):
                if k == y:
                    continue

                self.model.zero_grad()
                if x_adv.grad is not None:
                    x_adv.grad.zero_()

                f[0, k].backward(retain_graph=True)
                gradients.append(x_adv.grad.clone())

            # Find minimal perturbation
            with torch.no_grad():
                min_dist = float('inf')
                best_w = None
                best_f = None

                f_y = f[0, y].item()

                for k, grad_k in enumerate(gradients):
                    if k >= y:
                        k_class = k + 1
                    else:
                        k_class = k

                    w_k = grad_k
                    f_k = f[0, k_class].item()

                    # Distance to hyperplane
                    dist = abs(f_k - f_y) / (torch.norm(w_k.flatten()) + 1e-8)

                    if dist < min_dist:
                        min_dist = dist
                        best_w = w_k
                        best_f = f_k - f_y

                # Update adversarial example
                if best_w is not None:
                    r = (best_f / (torch.norm(best_w.flatten()) ** 2 + 1e-8)) * best_w
                    x_adv = x_adv + (1 + self.overshoot) * r

                    # Clip
                    if self.clip_min is not None and self.clip_max is not None:
                        x_adv = torch.clamp(x_adv, self.clip_min, self.clip_max)

            # Check new prediction
            with torch.no_grad():
                f = self.model(x_adv)
                pred_label = f.argmax(dim=1).item()

            iteration += 1

        # Compute metrics
        perturbation = (x_adv - x).detach()
        l2_norm = torch.norm(perturbation.flatten(), p=2).item()

        metrics = {
            'iterations': iteration,
            'l2_norm': l2_norm,
            'success': pred_label != y
        }

        return x_adv.squeeze(0).detach(), metrics

    def generate(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial examples for a batch.

        Args:
            x: Input tensor (batch, ...)
            y: True labels

        Returns:
            Tuple of (adversarial examples, metrics dict)
        """
        batch_size = x.size(0)
        x_adv_batch = []
        all_iterations = []
        all_l2_norms = []

        for i in range(batch_size):
            x_i = x[i]
            y_i = y[i].item()

            x_adv_i, metrics_i = self.generate_single(x_i, y_i)

            x_adv_batch.append(x_adv_i)
            all_iterations.append(metrics_i['iterations'])
            all_l2_norms.append(metrics_i['l2_norm'])

        x_adv = torch.stack(x_adv_batch, dim=0)

        # Aggregate metrics
        perturbation = (x_adv - x).detach()
        l2_norm = torch.norm(perturbation.view(batch_size, -1), p=2, dim=1).mean().item()
        linf_norm = torch.norm(perturbation.view(batch_size, -1), p=float('inf'), dim=1).mean().item()

        try:
            snr = compute_snr(x.detach(), x_adv.detach())
        except:
            snr = None

        metrics = {
            'l2_norm': l2_norm,
            'linf_norm': linf_norm,
            'snr_db': snr,
            'avg_iterations': np.mean(all_iterations),
            'max_iter': self.max_iter
        }

        return x_adv, metrics

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


def deepfool_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = 2,
    max_iter: int = 50
) -> torch.Tensor:
    """
    Simple DeepFool attack function.

    Args:
        model: Target model
        x: Input tensor
        y: Labels
        num_classes: Number of classes
        max_iter: Maximum iterations

    Returns:
        Adversarial examples
    """
    attack = DeepFool(model=model, num_classes=num_classes, max_iter=max_iter)
    x_adv, _ = attack.generate(x, y)
    return x_adv
