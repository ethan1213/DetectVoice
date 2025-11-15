"""
Metrics and visualization utilities for DetectVoice.
Automatically generates plots and reports.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report
)
import json
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class MetricsLogger:
    """
    Comprehensive metrics logger for DetectVoice models.
    Automatically generates all required plots and reports.
    """

    def __init__(self, model_name: str, save_dir: Path):
        """
        Initialize metrics logger.

        Args:
            model_name: Name of the model
            save_dir: Directory to save plots and reports
        """
        self.model_name = model_name
        self.save_dir = Path(save_dir) / model_name
        self.plots_dir = self.save_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epochs = []

        logger.info(f"MetricsLogger initialized for {model_name}")
        logger.info(f"Plots will be saved to: {self.plots_dir}")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float
    ) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            train_loss: Training loss
            val_loss: Validation loss
            train_acc: Training accuracy
            val_acc: Validation accuracy
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)

    def plot_loss_curve(self) -> None:
        """Plot and save loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.epochs, self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training and Validation Loss - {self.model_name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        save_path = self.plots_dir / 'loss_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Loss curve saved: {save_path}")

    def plot_accuracy_curve(self) -> None:
        """Plot and save accuracy curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_accs, label='Train Accuracy', linewidth=2)
        plt.plot(self.epochs, self.val_accs, label='Val Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Training and Validation Accuracy - {self.model_name}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        save_path = self.plots_dir / 'accuracy_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Accuracy curve saved: {save_path}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None
    ) -> None:
        """
        Plot and save confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
        """
        if labels is None:
            labels = ['Fake', 'Real']

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14)

        save_path = self.plots_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Confusion matrix saved: {save_path}")

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> None:
        """
        Plot and save ROC curve.

        Args:
            y_true: True labels
            y_scores: Prediction scores (probabilities)
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', linewidth=2, color='darkorange')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14)
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, alpha=0.3)

        save_path = self.plots_dir / 'roc_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ ROC curve saved: {save_path}")

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray
    ) -> None:
        """
        Plot and save Precision-Recall curve.

        Args:
            y_true: True labels
            y_scores: Prediction scores
        """
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}', linewidth=2, color='green')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', fontsize=14)
        plt.legend(fontsize=10, loc='lower left')
        plt.grid(True, alpha=0.3)

        save_path = self.plots_dir / 'precision_recall_curve.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Precision-Recall curve saved: {save_path}")

    def plot_spectrogram_comparison(
        self,
        real_spec: np.ndarray,
        fake_spec: np.ndarray
    ) -> None:
        """
        Plot comparison of real vs fake spectrograms.

        Args:
            real_spec: Real audio spectrogram
            fake_spec: Fake audio spectrogram
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Real spectrogram
        im1 = axes[0].imshow(real_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Real Audio Spectrogram', fontsize=12)
        axes[0].set_xlabel('Time', fontsize=10)
        axes[0].set_ylabel('Frequency', fontsize=10)
        plt.colorbar(im1, ax=axes[0], label='Magnitude (dB)')

        # Fake spectrogram
        im2 = axes[1].imshow(fake_spec, aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title('Fake Audio Spectrogram', fontsize=12)
        axes[1].set_xlabel('Time', fontsize=10)
        axes[1].set_ylabel('Frequency', fontsize=10)
        plt.colorbar(im2, ax=axes[1], label='Magnitude (dB)')

        plt.suptitle(f'Real vs Fake Comparison - {self.model_name}', fontsize=14, y=1.02)
        plt.tight_layout()

        save_path = self.plots_dir / 'spectrogram_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Spectrogram comparison saved: {save_path}")

    def generate_all_plots(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        real_spec: Optional[np.ndarray] = None,
        fake_spec: Optional[np.ndarray] = None
    ) -> None:
        """
        Generate all plots at once.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores
            real_spec: Example real spectrogram (optional)
            fake_spec: Example fake spectrogram (optional)
        """
        logger.info("Generating all plots...")

        # Training curves
        if len(self.epochs) > 0:
            self.plot_loss_curve()
            self.plot_accuracy_curve()

        # Evaluation plots
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_scores)
        self.plot_precision_recall_curve(y_true, y_scores)

        # Spectrogram comparison
        if real_spec is not None and fake_spec is not None:
            self.plot_spectrogram_comparison(real_spec, fake_spec)

        logger.info("✓ All plots generated successfully")

    def save_metrics_json(self, metrics: dict) -> None:
        """
        Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
        """
        json_path = self.save_dir / 'metrics.json'

        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"✓ Metrics saved to: {json_path}")

    def save_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Save classification report to text file.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        report_path = self.save_dir / 'classification_report.txt'

        report = classification_report(
            y_true,
            y_pred,
            target_names=['Fake', 'Real'],
            zero_division=0
        )

        with open(report_path, 'w') as f:
            f.write(f"Classification Report - {self.model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report)

        logger.info(f"✓ Classification report saved: {report_path}")
