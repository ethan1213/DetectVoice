"""
Adversarial evaluation module for DetectVoice.
Tests model robustness against various attacks.

⚠️ FOR DEFENSIVE RESEARCH ONLY
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import json
import logging
import matplotlib.pyplot as plt
import numpy as np

from src.adversarial.fgsm import FGSM
from src.adversarial.pgd import PGD

logger = logging.getLogger(__name__)


class AdversarialEvaluator:
    """
    Comprehensive adversarial evaluation for DetectVoice models.
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: str = 'cpu',
        save_dir: Optional[Path] = None
    ):
        """
        Initialize adversarial evaluator.

        Args:
            model: Model to evaluate
            model_name: Name of the model
            device: Device to use
            save_dir: Directory to save reports
        """
        self.model = model
        self.model_name = model_name
        self.device = device

        if save_dir is None:
            save_dir = Path("reports/adversarial")

        self.save_dir = Path(save_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Adversarial evaluator initialized for {model_name}")
        logger.info(f"Reports will be saved to: {self.save_dir}")

    def evaluate_all_attacks(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Evaluate model against all adversarial attacks.

        Args:
            dataloader: Data loader with test samples

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info(f"ADVERSARIAL EVALUATION - {self.model_name}")
        logger.info("=" * 60)

        results = {}

        # Get first batch for testing
        inputs, labels = next(iter(dataloader))
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # 1. FGSM Attack
        logger.info("\n[1/2] Testing FGSM attack...")
        fgsm = FGSM(self.model, epsilon=0.03, device=self.device)
        results['fgsm'] = fgsm.evaluate(inputs, labels)

        logger.info(f"  Clean Accuracy: {results['fgsm']['clean_accuracy']:.4f}")
        logger.info(f"  Adversarial Accuracy: {results['fgsm']['adversarial_accuracy']:.4f}")
        logger.info(f"  Robustness Drop: {results['fgsm']['robustness_drop']:.4f}")

        # 2. PGD Attack
        logger.info("\n[2/2] Testing PGD attack...")
        pgd = PGD(self.model, epsilon=0.03, alpha=0.01, num_iter=10, device=self.device)
        results['pgd'] = pgd.evaluate(inputs, labels)

        logger.info(f"  Clean Accuracy: {results['pgd']['clean_accuracy']:.4f}")
        logger.info(f"  Adversarial Accuracy: {results['pgd']['adversarial_accuracy']:.4f}")
        logger.info(f"  Robustness Drop: {results['pgd']['robustness_drop']:.4f}")

        # Save results
        self._save_results(results)

        # Generate plots
        self._plot_results(results)

        logger.info("\n" + "=" * 60)
        logger.info("✓ ADVERSARIAL EVALUATION COMPLETE")
        logger.info("=" * 60)

        return results

    def _save_results(self, results: Dict) -> None:
        """Save results to JSON."""
        json_path = self.save_dir / "adversarial_results.json"

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"\n✓ Results saved to: {json_path}")

    def _plot_results(self, results: Dict) -> None:
        """Generate visualization of adversarial results."""
        attacks = list(results.keys())
        clean_accs = [results[a]['clean_accuracy'] for a in attacks]
        adv_accs = [results[a]['adversarial_accuracy'] for a in attacks]

        x = np.arange(len(attacks))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, clean_accs, width, label='Clean', color='green', alpha=0.8)
        bars2 = ax.bar(x + width/2, adv_accs, width, label='Adversarial', color='red', alpha=0.8)

        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'Adversarial Robustness - {self.model_name}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([a.upper() for a in attacks])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2%}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        plot_path = self.save_dir / "adversarial_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Plot saved to: {plot_path}")

    def generate_adversarial_examples(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        attack_type: str = 'fgsm'
    ) -> torch.Tensor:
        """
        Generate adversarial examples for visualization.

        Args:
            inputs: Input tensors
            labels: Labels
            attack_type: Type of attack ('fgsm' or 'pgd')

        Returns:
            Adversarial examples
        """
        if attack_type == 'fgsm':
            attack = FGSM(self.model, epsilon=0.03, device=self.device)
        elif attack_type == 'pgd':
            attack = PGD(self.model, epsilon=0.03, alpha=0.01, num_iter=10, device=self.device)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        adv_inputs, _ = attack.generate(inputs, labels)

        return adv_inputs

    def visualize_attack(
        self,
        clean_input: torch.Tensor,
        adv_input: torch.Tensor,
        attack_name: str = 'FGSM'
    ) -> None:
        """
        Visualize clean vs adversarial input.

        Args:
            clean_input: Clean input
            adv_input: Adversarial input
            attack_name: Name of attack
        """
        # Convert to numpy
        clean_np = clean_input.squeeze().cpu().numpy()
        adv_np = adv_input.squeeze().cpu().numpy()
        perturbation = (adv_np - clean_np)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Clean
        im1 = axes[0].imshow(clean_np, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Clean Input', fontsize=12)
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        plt.colorbar(im1, ax=axes[0])

        # Adversarial
        im2 = axes[1].imshow(adv_np, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(f'{attack_name} Adversarial', fontsize=12)
        axes[1].set_xlabel('Time')
        plt.colorbar(im2, ax=axes[1])

        # Perturbation
        im3 = axes[2].imshow(perturbation, aspect='auto', origin='lower', cmap='RdBu_r')
        axes[2].set_title('Perturbation', fontsize=12)
        axes[2].set_xlabel('Time')
        plt.colorbar(im3, ax=axes[2])

        plt.suptitle(f'{attack_name} Attack Visualization - {self.model_name}', fontsize=14, y=1.02)
        plt.tight_layout()

        save_path = self.save_dir / f"{attack_name.lower()}_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Visualization saved: {save_path}")
