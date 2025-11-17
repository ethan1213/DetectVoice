"""
Master Training Script with MLflow and TensorBoard Integration
Supports all model types with comprehensive logging
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import os

class MasterTrainer:
    """Universal trainer for all model types"""
    def __init__(self, model, config: Dict, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.writer = None
        self.mlflow_run = None

        # Setup logging
        if config.get('tensorboard', {}).get('enabled', True):
            self.writer = SummaryWriter(config.get('tensorboard', {}).get('log_dir', 'outputs/logs/tensorboard'))

        if config.get('mlflow', {}).get('enabled', True):
            mlflow.set_tracking_uri(config.get('mlflow', {}).get('tracking_uri', 'outputs/logs/mlruns'))
            mlflow.set_experiment(config.get('mlflow', {}).get('experiment_name', 'detectvoice'))

        # Optimizer
        self.optimizer = self._get_optimizer()

        # Scheduler
        self.scheduler = self._get_scheduler()

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Best metrics
        self.best_metric = 0.0

    def _get_optimizer(self):
        opt_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 0.0001)
        weight_decay = self.config.get('weight_decay', 0.01)

        if opt_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    def _get_scheduler(self):
        sched_cfg = self.config.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'cosine_annealing')

        if sched_type == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.get('epochs', 100))
        elif sched_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5)
        else:
            return None

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            if self.config.get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip_val'])

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

        return total_loss / len(train_loader), 100. * correct / total

    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for inputs, labels in val_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = 100. * correct / total
        return total_loss / len(val_loader), acc, np.array(all_preds), np.array(all_labels)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        """Main training loop"""
        if self.config.get('mlflow', {}).get('enabled', True):
            mlflow.start_run()
            mlflow.log_params(self.config)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)

            print(f'\nEpoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # MLflow logging
            if self.config.get('mlflow', {}).get('enabled', True):
                mlflow.log_metrics({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}, step=epoch)

            # Save best model
            if val_acc > self.best_metric:
                self.best_metric = val_acc
                self.save_checkpoint(epoch, val_acc, 'best')

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Early stopping
            # (implement if needed)

        if self.config.get('mlflow', {}).get('enabled', True):
            mlflow.end_run()

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, epoch: int, metric: float, name: str = 'checkpoint'):
        os.makedirs('outputs/models', exist_ok=True)
        path = f'outputs/models/{name}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric': metric,
            'config': self.config
        }, path)
        print(f'Saved checkpoint to {path}')
