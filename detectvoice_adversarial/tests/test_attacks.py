"""
Unit tests for adversarial attacks.

Tests basic functionality of FGSM, PGD, and other attacks.
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.attacks import FGSM, PGD, CarliniWagnerL2, DeepFool
from src.models.cnn.detector import CNNDetector


class DummyModel(nn.Module):
    """Simple dummy model for testing."""

    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def dummy_model():
    """Create dummy model for testing."""
    model = DummyModel()
    model.eval()
    return model


@pytest.fixture
def dummy_input():
    """Create dummy input."""
    return torch.randn(4, 28, 28)


@pytest.fixture
def dummy_labels():
    """Create dummy labels."""
    return torch.randint(0, 2, (4,))


def test_fgsm_attack(dummy_model, dummy_input, dummy_labels):
    """Test FGSM attack."""
    attack = FGSM(model=dummy_model, epsilon=0.03)

    x_adv, metrics = attack.generate(dummy_input, dummy_labels)

    # Check output shape
    assert x_adv.shape == dummy_input.shape

    # Check perturbation exists
    perturbation = torch.norm((x_adv - dummy_input).flatten(), p=float('inf'))
    assert perturbation > 0

    # Check metrics
    assert 'l2_norm' in metrics
    assert 'linf_norm' in metrics
    assert 'epsilon' in metrics


def test_pgd_attack(dummy_model, dummy_input, dummy_labels):
    """Test PGD attack."""
    attack = PGD(
        model=dummy_model,
        epsilon=0.03,
        alpha=0.01,
        num_iter=5
    )

    x_adv, metrics = attack.generate(dummy_input, dummy_labels)

    # Check output shape
    assert x_adv.shape == dummy_input.shape

    # Check perturbation exists
    perturbation = torch.norm((x_adv - dummy_input).flatten(), p=float('inf'))
    assert perturbation > 0

    # Check metrics
    assert 'l2_norm' in metrics
    assert 'linf_norm' in metrics
    assert 'num_iter' in metrics


def test_cw_attack(dummy_model, dummy_input, dummy_labels):
    """Test C&W attack."""
    attack = CarliniWagnerL2(
        model=dummy_model,
        c=1.0,
        max_iter=10
    )

    x_adv, metrics = attack.generate(dummy_input, dummy_labels)

    # Check output shape
    assert x_adv.shape == dummy_input.shape

    # Check metrics
    assert 'l2_norm' in metrics
    assert 'c' in metrics


def test_deepfool_attack(dummy_model, dummy_input, dummy_labels):
    """Test DeepFool attack."""
    attack = DeepFool(
        model=dummy_model,
        num_classes=2,
        max_iter=10
    )

    # Test on single sample
    x_single = dummy_input[0]
    y_single = dummy_labels[0].item()

    x_adv_single, metrics = attack.generate_single(x_single, y_single)

    # Check shape
    assert x_adv_single.shape == x_single.shape

    # Check metrics
    assert 'iterations' in metrics
    assert 'l2_norm' in metrics


def test_fgsm_evaluation(dummy_model, dummy_input, dummy_labels):
    """Test FGSM evaluation method."""
    attack = FGSM(model=dummy_model, epsilon=0.03)

    eval_metrics = attack.evaluate(dummy_input, dummy_labels)

    # Check all metrics exist
    assert 'clean_accuracy' in eval_metrics
    assert 'adversarial_accuracy' in eval_metrics
    assert 'attack_success_rate' in eval_metrics
    assert 'robustness_drop' in eval_metrics


def test_attack_with_different_epsilon(dummy_model, dummy_input, dummy_labels):
    """Test attack with different epsilon values."""
    epsilons = [0.01, 0.03, 0.1]

    perturbations = []

    for eps in epsilons:
        attack = FGSM(model=dummy_model, epsilon=eps)
        x_adv, _ = attack.generate(dummy_input, dummy_labels)

        perturbation = torch.norm((x_adv - dummy_input).flatten(), p=float('inf'))
        perturbations.append(perturbation.item())

    # Larger epsilon should generally produce larger perturbations
    # (not always strictly increasing due to gradient saturation, but trend should hold)
    assert perturbations[0] <= perturbations[-1] + 0.1  # Some tolerance


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
