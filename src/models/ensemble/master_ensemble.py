"""
Master Ensemble Model
Unifies multiple fusion strategies (averaging, jury, stacking) into a single,
PyTorch-compatible module that can be used with the existing training pipeline.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tqdm import tqdm
import joblib
from pathlib import Path

class EnsembleModel(nn.Module):
    """
    A unified ensemble model that supports various strategies.

    Args:
        base_models (Dict[str, nn.Module]): A dictionary of pre-trained base models.
        config (Dict): The 'ensemble' section of the main config file.
    """
    def __init__(self, base_models: Dict[str, nn.Module], config: Dict):
        super().__init__()
        # Ensure base models are not trainable within the ensemble
        self.base_models = nn.ModuleDict({name: model.eval() for name, model in base_models.items()})
        self.config = config
        self.strategy = self.config.get('strategy', 'averaging')
        
        self.meta_model = None
        if self.strategy == 'stacking':
            meta_model_type = self.config.get('stacking', {}).get('meta_model', 'xgboost')
            if meta_model_type == 'xgboost':
                self.meta_model = XGBClassifier(**self.config.get('xgboost_params', {}))
            elif meta_model_type == 'logistic':
                self.meta_model = LogisticRegression(**self.config.get('logistic_params', {}))
            else:
                raise ValueError(f"Unsupported meta-model type: {meta_model_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single input tensor `x`.
        Assumes all base models take the same input tensor.
        """
        # Get predictions (logits) from all base models
        # We detach them as we don't want to backpropagate through base models
        all_logits = [model(x).detach() for model in self.base_models.values()]
        
        if self.strategy == 'averaging':
            return torch.stack(all_logits).mean(dim=0)
        
        elif self.strategy == 'weighted_voting':
            weights = self.config.get('weights', {})
            # Ensure weights are in the same order as models
            weight_tensor = torch.tensor([weights.get(name, 1.0) for name in self.base_models.keys()], device=x.device)
            weight_tensor /= weight_tensor.sum() # Normalize
            
            # Reshape for broadcasting: (num_models, 1, 1)
            weight_tensor = weight_tensor.view(-1, 1, 1)
            
            # Weighted sum of logits
            return (torch.stack(all_logits, dim=0) * weight_tensor).sum(dim=0)

        elif self.strategy == 'jury':
            return self._jury_predict(all_logits)
            
        elif self.strategy == 'stacking':
            if self.meta_model is None or not hasattr(self.meta_model, '_Booster'):
                raise RuntimeError("Stacking meta-model has not been trained. Run `train_meta_model` first.")
            
            # Use probabilities for stacking features
            if self.config.get('stacking', {}).get('use_proba', True):
                stacked_features = torch.cat([torch.softmax(logits, dim=1) for logits in all_logits], dim=1)
            else: # use logits
                stacked_features = torch.cat(all_logits, dim=1)
            
            # Predict with the scikit-learn meta-model
            meta_preds_proba = self.meta_model.predict_proba(stacked_features.cpu().numpy())
            return torch.from_numpy(meta_preds_proba).to(x.device)

        else:
            raise ValueError(f"Unknown ensemble strategy: {self.strategy}")

    def _jury_predict(self, all_logits: List[torch.Tensor]):
        """Jury prediction logic, returns the most voted class logits."""
        all_preds = torch.stack([torch.argmax(logits, dim=1) for logits in all_logits])
        # Find the most frequent prediction for each item in the batch
        majority_vote, _ = torch.mode(all_preds, dim=0)
        # Convert back to one-hot encoding-like logits for compatibility with loss functions
        return nn.functional.one_hot(majority_vote, num_classes=all_logits[0].shape[1]).float()

    def train_meta_model(self, data_loader: torch.utils.data.DataLoader, models_dir: str):
        """
        Trains the stacking meta-model.
        
        Args:
            data_loader: DataLoader for the training set.
            models_dir: Directory to save the trained meta-model.
        """
        if self.strategy != 'stacking':
            print("Meta-model training is only applicable for the 'stacking' strategy.")
            return
            
        print("--- Training Stacking Meta-Model ---")
        
        all_base_preds = []
        all_labels = []
        
        print("Step 1: Generating predictions from base models...")
        for inputs, labels in tqdm(data_loader):
            # Assuming inputs is a tensor, not a dict
            inputs = inputs.to(next(self.parameters()).device)
            
            # Get predictions from base models
            with torch.no_grad():
                if self.config.get('stacking', {}).get('use_proba', True):
                    # Use probabilities as features
                    preds = [torch.softmax(model(inputs), dim=1) for model in self.base_models.values()]
                else:
                    # Use logits as features
                    preds = [model(inputs) for model in self.base_models.values()]
            
            # Concatenate predictions for the batch and store
            batch_preds = torch.cat(preds, dim=1)
            all_base_preds.append(batch_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Combine all batches
        X_meta = np.concatenate(all_base_preds, axis=0)
        y_meta = np.concatenate(all_labels, axis=0)
        
        print(f"Step 2: Fitting meta-model on {X_meta.shape[0]} samples...")
        self.meta_model.fit(X_meta, y_meta)
        
        # Save the trained meta-model
        save_path = Path(models_dir) / "ensemble_meta_model.joblib"
        joblib.dump(self.meta_model, save_path)
        print(f"--- Meta-Model trained and saved to {save_path} ---")

    def load_meta_model(self, models_dir: str):
        """Loads a pre-trained meta-model from disk."""
        if self.strategy != 'stacking':
            return
        
        load_path = Path(models_dir) / "ensemble_meta_model.joblib"
        if not load_path.exists():
            raise FileNotFoundError(f"Meta-model not found at {load_path}. Please train it first.")
            
        self.meta_model = joblib.load(load_path)
        print(f"Stacking meta-model loaded from {load_path}")