"""
Robustness evaluation suite for adversarial testing.

Evaluates detector models against:
- Clean samples
- Synthetic (fake) samples
- Adversarial examples (FGSM, PGD, CW, DeepFool)
- Spectral perturbations
- GAN-generated samples (toy)

⚠️  SECURITY & ETHICS NOTICE ⚠️
This evaluation is for DEFENSIVE research to improve detector robustness.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.attacks import FGSM, PGD, CarliniWagnerL2, DeepFool
from src.attacks.spec_perturbations import SpectralPerturbation, LowAmplitudeNoise
from src.utils.logger import get_logger
from src.utils.audio import compute_snr

logger = get_logger(__name__)


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation for deepfake detectors.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        save_dir: Optional[Path] = None
    ):
        """
        Initialize robustness evaluator.

        Args:
            model: Detector model to evaluate
            device: Device to use
            save_dir: Directory to save results
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("artifacts/robustness")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Robustness evaluator initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Results will be saved to: {self.save_dir}")

    def evaluate_clean(
        self,
        dataloader: DataLoader
    ) -> Dict:
        """
        Evaluate on clean samples.

        Args:
            dataloader: DataLoader with test samples

        Returns:
            Dictionary with metrics
        """
        logger.info("Evaluating on CLEAN samples...")

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        metrics = self._compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        logger.info(f"Clean Accuracy: {metrics['accuracy']:.4f}")
        return metrics

    def evaluate_adversarial(
        self,
        dataloader: DataLoader,
        attack_name: str,
        attack_params: Dict
    ) -> Dict:
        """
        Evaluate on adversarial examples.

        Args:
            dataloader: DataLoader with test samples
            attack_name: Name of attack ('fgsm', 'pgd', 'cw', 'deepfool')
            attack_params: Attack parameters

        Returns:
            Dictionary with metrics
        """
        logger.info(f"Evaluating on ADVERSARIAL samples ({attack_name.upper()})...")

        # Initialize attack
        attack = self._get_attack(attack_name, attack_params)

        all_preds = []
        all_labels = []
        all_probs = []
        all_snrs = []

        loss_fn = nn.CrossEntropyLoss()

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Generate adversarial examples
            if hasattr(attack, 'generate'):
                inputs_adv, attack_metrics = attack.generate(inputs, labels, loss_fn)
                if 'snr_db' in attack_metrics and attack_metrics['snr_db'] is not None:
                    all_snrs.append(attack_metrics['snr_db'])
            else:
                inputs_adv = attack(inputs, labels)

            # Evaluate
            with torch.no_grad():
                outputs = self.model(inputs_adv)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        metrics = self._compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )

        if all_snrs:
            metrics['avg_snr_db'] = np.mean(all_snrs)

        logger.info(f"{attack_name.upper()} Adversarial Accuracy: {metrics['accuracy']:.4f}")

        return metrics

    def evaluate_comprehensive(
        self,
        dataloader: DataLoader,
        attacks_config: Optional[Dict] = None
    ) -> Dict:
        """
        Comprehensive evaluation across multiple attack types.

        Args:
            dataloader: DataLoader with test samples
            attacks_config: Configuration for attacks (optional)

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE ROBUSTNESS EVALUATION")
        logger.info("=" * 60)

        results = {}

        # 1. Clean evaluation
        results['clean'] = self.evaluate_clean(dataloader)

        # 2. Default attack configurations
        if attacks_config is None:
            attacks_config = {
                'fgsm': {'epsilon': 0.03},
                'pgd': {'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 10},
                'cw': {'c': 1.0, 'max_iter': 50},
                'deepfool': {'num_classes': 2, 'max_iter': 20}
            }

        # 3. Adversarial evaluations
        for attack_name, attack_params in attacks_config.items():
            try:
                results[f'adv_{attack_name}'] = self.evaluate_adversarial(
                    dataloader,
                    attack_name,
                    attack_params
                )
            except Exception as e:
                logger.error(f"Error evaluating {attack_name}: {e}")
                results[f'adv_{attack_name}'] = {'error': str(e)}

        # 4. Compute robustness drops
        clean_acc = results['clean']['accuracy']
        for key, value in results.items():
            if key.startswith('adv_') and 'accuracy' in value:
                value['robustness_drop'] = clean_acc - value['accuracy']

        # 5. Save results
        self._save_results(results)

        logger.info("=" * 60)
        logger.info("✓ EVALUATION COMPLETE")
        logger.info("=" * 60)

        return results

    def _get_attack(self, attack_name: str, params: Dict):
        """Get attack instance based on name."""
        attack_name = attack_name.lower()

        if attack_name == 'fgsm':
            return FGSM(model=self.model, **params)
        elif attack_name == 'pgd':
            return PGD(model=self.model, **params)
        elif attack_name == 'cw':
            return CarliniWagnerL2(model=self.model, **params)
        elif attack_name == 'deepfool':
            return DeepFool(model=self.model, **params)
        else:
            raise ValueError(f"Unknown attack: {attack_name}")

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Compute classification metrics."""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        }

        try:
            metrics['auroc'] = float(roc_auc_score(y_true, y_proba))
        except:
            metrics['auroc'] = 0.0

        return metrics

    def _save_results(self, results: Dict):
        """Save results to JSON and CSV."""
        # Save JSON
        json_path = self.save_dir / "robustness_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)

        logger.info(f"✓ Results saved to: {json_path}")

        # Save summary CSV
        csv_path = self.save_dir / "robustness_summary.csv"
        with open(csv_path, 'w') as f:
            f.write("Attack,Accuracy,Precision,Recall,F1,AUROC,Robustness Drop\n")

            for attack, metrics in results.items():
                if 'accuracy' in metrics:
                    f.write(f"{attack},"
                           f"{metrics.get('accuracy', 0.0):.4f},"
                           f"{metrics.get('precision', 0.0):.4f},"
                           f"{metrics.get('recall', 0.0):.4f},"
                           f"{metrics.get('f1_score', 0.0):.4f},"
                           f"{metrics.get('auroc', 0.0):.4f},"
                           f"{metrics.get('robustness_drop', 0.0):.4f}\n")

        logger.info(f"✓ Summary saved to: {csv_path}")
