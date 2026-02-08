"""
Master Training Script with MLflow and TensorBoard Integration
Supports all model types with comprehensive logging, early stopping, and advanced metrics.
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MasterTrainer:
    """
    Universal trainer for all model types with early stopping and advanced metrics.
    """
    def __init__(self, model, config: Dict, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.training_config = self.config.get('training', {})
        self.logging_config = self.config.get('logging', {})
        
        self.writer = None
        self.mlflow_run = None

        # Setup logging
        if self.logging_config.get('tensorboard', {}).get('enabled', True):
            self.writer = SummaryWriter(self.logging_config.get('tensorboard', {}).get('log_dir', 'outputs/logs/tensorboard'))

        if self.logging_config.get('mlflow', {}).get('enabled', True):
            mlflow.set_tracking_uri(self.logging_config.get('mlflow', {}).get('tracking_uri', 'outputs/logs/mlruns'))
            mlflow.set_experiment(self.logging_config.get('mlflow', {}).get('experiment_name', 'detectvoice'))

        # Optimizer, Scheduler, and Loss
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        # Early Stopping
        self.early_stopping_config = self.training_config.get('early_stopping', {'enabled': False})
        self.es_patience = self.early_stopping_config.get('patience', 10)
        self.es_min_delta = self.early_stopping_config.get('min_delta', 0.001)
        self.es_monitor = self.early_stopping_config.get('monitor', 'val_loss')
        self.es_counter = 0
        self.es_best_value = -np.inf if self.es_monitor != 'val_loss' else np.inf

        # Best metrics for checkpointing
        self.checkpoint_config = self.training_config.get('checkpoint', {})
        self.best_checkpoint_metric = 0.0

    def _get_optimizer(self):
        opt_name = self.training_config.get('optimizer', 'adamw').lower()
        lr = self.training_config.get('learning_rate', 0.0001)
        weight_decay = self.training_config.get('weight_decay', 0.01)

        if opt_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    def _get_scheduler(self):
        sched_cfg = self.training_config.get('scheduler', {})
        sched_type = sched_cfg.get('type', 'cosine_annealing')

        if sched_type == 'cosine_annealing':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.training_config.get('epochs', 100))
        elif sched_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=5, factor=0.5)
        else:
            return None

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Training]')
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.training_config.get('gradient_clip_val', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config['gradient_clip_val'])

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{total_loss / (pbar.n + 1):.4f}', 'acc': f'{100.*total_correct/total_samples:.2f}%'})

        return total_loss / len(train_loader), 100. * total_correct / total_samples

    @torch.no_grad()
    def validate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(val_loader, desc='Validating')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            total_correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate all metrics
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'val_acc': accuracy_score(all_labels, all_preds) * 100,
            'val_f1': f1_score(all_labels, all_preds, average='binary') * 100,
            'val_precision': precision_score(all_labels, all_preds, average='binary') * 100,
            'val_recall': recall_score(all_labels, all_preds, average='binary') * 100,
        }
        return metrics

    def _check_early_stopping(self, current_metric_value: float) -> bool:
        if not self.early_stopping_config.get('enabled', False):
            return False

        should_stop = False
        # Adjust for 'loss' where lower is better
        if self.es_monitor.endswith('loss'):
            if self.es_best_value - current_metric_value > self.es_min_delta:
                self.es_best_value = current_metric_value
                self.es_counter = 0
            else:
                self.es_counter += 1
        # For metrics where higher is better (acc, f1, etc.)
        else:
            if current_metric_value - self.es_best_value > self.es_min_delta:
                self.es_best_value = current_metric_value
                self.es_counter = 0
            else:
                self.es_counter += 1
        
        if self.es_counter >= self.es_patience:
            print(f"\n--- Early stopping triggered after {self.es_patience} epochs with no improvement ---")
            should_stop = True
            
        return should_stop

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs = self.training_config.get('epochs', 100)
        if self.logging_config.get('mlflow', {}).get('enabled', True):
            mlflow.start_run()
            mlflow.log_params(self.config)

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.2f}%, F1: {val_metrics['val_f1']:.2f}%")

            # Logging
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
                for key, val in val_metrics.items():
                    self.writer.add_scalar(f"Validation/{key.replace('_', ' ').title()}", val, epoch)

            if self.logging_config.get('mlflow', {}).get('enabled', True):
                mlflow.log_metrics({'train_loss': train_loss, 'train_acc': train_acc, **val_metrics}, step=epoch)

            # Checkpoint best model
            monitor_metric = self.checkpoint_config.get('monitor', 'val_f1')
            if val_metrics.get(monitor_metric, 0) > self.best_checkpoint_metric:
                self.best_checkpoint_metric = val_metrics[monitor_metric]
                self.save_checkpoint(epoch, self.best_checkpoint_metric, 'best_model')
                print(f"✅ New best model saved with {monitor_metric}: {self.best_checkpoint_metric:.2f}%")

            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.es_monitor])
                else:
                    self.scheduler.step()

            # Early stopping check
            if self._check_early_stopping(val_metrics[self.es_monitor]):
                break

        if self.logging_config.get('mlflow', {}).get('enabled', True):
            mlflow.end_run()

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, epoch: int, metric: float, name: str):
        path = Path(self.config.get('paths', {}).get('models_dir', 'outputs/models'))
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f'{name}.pt'
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metric_value': metric,
            'metric_name': self.checkpoint_config.get('monitor', 'val_f1'),
            'config': self.config
        }, file_path)