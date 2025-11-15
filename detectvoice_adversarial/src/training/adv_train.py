"""
Adversarial Training for robust deepfake detection.

Trains models with adversarial examples to improve robustness.

⚠️  SECURITY & ETHICS NOTICE ⚠️
Adversarial training is for DEFENSIVE purposes:
improving detector robustness against attacks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm

from src.attacks import FGSM, PGD
from src.utils.logger import get_logger, setup_logger
from src.export.export_utils import ModelExporter

logger = get_logger(__name__)


class AdversarialTrainer:
    """
    Adversarial training framework for robust model training.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        adv_ratio: float = 0.5,
        attack_type: str = 'pgd',
        attack_params: Optional[Dict] = None,
        save_dir: Optional[Path] = None
    ):
        """
        Initialize adversarial trainer.

        Args:
            model: Model to train
            optimizer: Optimizer
            device: Device to use
            adv_ratio: Ratio of adversarial examples in each batch
            attack_type: Type of attack ('fgsm' or 'pgd')
            attack_params: Attack parameters
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.adv_ratio = adv_ratio
        self.attack_type = attack_type
        self.save_dir = Path(save_dir) if save_dir else Path("artifacts/models")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Setup attack
        if attack_params is None:
            if attack_type == 'fgsm':
                attack_params = {'epsilon': 0.03}
            elif attack_type == 'pgd':
                attack_params = {'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 7}

        self.attack_params = attack_params

        logger.info(f"Adversarial trainer initialized")
        logger.info(f"Attack type: {attack_type}")
        logger.info(f"Adversarial ratio: {adv_ratio}")
        logger.info(f"Attack params: {attack_params}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: nn.Module,
        epoch: int
    ) -> Dict:
        """
        Train one epoch with adversarial examples.

        Args:
            train_loader: Training data loader
            loss_fn: Loss function
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            # Split batch into clean and adversarial
            num_adv = int(batch_size * self.adv_ratio)
            num_clean = batch_size - num_adv

            # Clean samples
            clean_inputs = inputs[:num_clean]
            clean_labels = labels[:num_clean]

            # Adversarial samples
            if num_adv > 0:
                adv_inputs_orig = inputs[num_clean:]
                adv_labels = labels[num_clean:]

                # Generate adversarial examples
                attack = self._get_attack()
                adv_inputs, _ = attack.generate(adv_inputs_orig, adv_labels, loss_fn)

                # Combine clean and adversarial
                all_inputs = torch.cat([clean_inputs, adv_inputs], dim=0)
                all_labels = torch.cat([clean_labels, adv_labels], dim=0)
            else:
                all_inputs = clean_inputs
                all_labels = clean_labels

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(all_inputs)
            loss = loss_fn(outputs, all_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == all_labels).sum().item()
            total += all_labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })

        metrics = {
            'train_loss': total_loss / len(train_loader),
            'train_acc': correct / total
        }

        return metrics

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module
    ) -> Dict:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            loss_fn: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Also evaluate on adversarial validation set
        correct_adv = 0
        total_adv = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Clean validation
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        # Adversarial validation (on subset)
        val_subset = list(val_loader)[:min(10, len(val_loader))]
        attack = self._get_attack()

        for inputs, labels in val_subset:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Generate adversarial examples
            adv_inputs, _ = attack.generate(inputs, labels, loss_fn)

            # Evaluate
            with torch.no_grad():
                outputs = self.model(adv_inputs)
                preds = outputs.argmax(dim=1)
                correct_adv += (preds == labels).sum().item()
                total_adv += labels.size(0)

        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': correct / total,
            'val_acc_adv': correct_adv / max(total_adv, 1)
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        loss_fn: Optional[nn.Module] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Full training loop with adversarial training.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            loss_fn: Loss function (default: CrossEntropyLoss)
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Early stopping patience

        Returns:
            Training history
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        logger.info("=" * 60)
        logger.info("ADVERSARIAL TRAINING")
        logger.info("=" * 60)

        best_val_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_acc_adv': []
        }

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader, loss_fn, epoch)

            # Validate
            val_metrics = self.validate(val_loader, loss_fn)

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])
            history['val_acc_adv'].append(val_metrics['val_acc_adv'])

            # Log
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f} | "
                       f"Train Acc: {train_metrics['train_acc']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f} | "
                       f"Val Acc (Clean): {val_metrics['val_acc']:.4f} | "
                       f"Val Acc (Adv): {val_metrics['val_acc_adv']:.4f}")

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Save best model
            if val_metrics['val_acc'] > best_val_acc:
                best_val_acc = val_metrics['val_acc']
                patience_counter = 0

                checkpoint_path = self.save_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'history': history
                }, checkpoint_path)

                logger.info(f"✓ Best model saved (Val Acc: {best_val_acc:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break

        logger.info("=" * 60)
        logger.info(f"✓ TRAINING COMPLETE")
        logger.info(f"Best Val Acc: {best_val_acc:.4f}")
        logger.info("=" * 60)

        return history

    def _get_attack(self):
        """Get attack instance."""
        if self.attack_type == 'fgsm':
            return FGSM(model=self.model, **self.attack_params)
        elif self.attack_type == 'pgd':
            return PGD(model=self.model, **self.attack_params)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
