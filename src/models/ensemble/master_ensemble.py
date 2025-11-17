"""
Master Ensemble combining all model types
Supports multiple fusion strategies
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
import warnings

class MasterEnsemble(nn.Module):
    """
    Master ensemble combining Transformers, Classical, Deep Learning, and Advanced models
    """
    def __init__(self, models: Dict[str, nn.Module], weights: Optional[Dict[str, float]] = None, strategy: str = "weighted_avg"):
        """
        Args:
            models: Dict of model_name -> model
            weights: Dict of model_name -> weight (for weighted strategies)
            strategy: 'avg', 'weighted_avg', 'max', 'stacking'
        """
        super().__init__()
        self.model_dict = nn.ModuleDict(models)
        self.strategy = strategy
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            inputs: Dict of model_name -> model_input (allows different inputs per model)
        """
        all_logits = []
        all_weights = []

        for name, model in self.model_dict.items():
            if name in inputs:
                logits = model(inputs[name])
                all_logits.append(logits)
                all_weights.append(self.weights.get(name, 1.0))

        if not all_logits:
            raise ValueError("No valid model outputs")

        all_logits = torch.stack(all_logits, dim=0)  # (num_models, batch, classes)
        all_weights = torch.tensor(all_weights, device=all_logits.device).view(-1, 1, 1)

        if self.strategy == "avg":
            return all_logits.mean(dim=0)
        elif self.strategy == "weighted_avg":
            all_weights = all_weights / all_weights.sum()
            return (all_logits * all_weights).sum(dim=0)
        elif self.strategy == "max":
            return all_logits.max(dim=0)[0]
        else:
            return all_logits.mean(dim=0)

class JuryEnsemble:
    """
    Jury system: requires N models to agree with confidence threshold
    """
    def __init__(self, models: Dict, min_agreement: int = 3, confidence_threshold: float = 0.7):
        self.models = models
        self.min_agreement = min_agreement
        self.confidence_threshold = confidence_threshold

    def predict(self, inputs: Dict[str, torch.Tensor]) -> tuple:
        """
        Returns: (predictions, agreement_count, is_confident)
        """
        all_probs = []
        all_preds = []

        for name, model in self.models.items():
            if name in inputs:
                with torch.no_grad():
                    logits = model(inputs[name])
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_probs.append(probs)
                    all_preds.append(preds)

        all_preds = torch.stack(all_preds, dim=0)  # (num_models, batch)
        all_probs = torch.stack(all_probs, dim=0)  # (num_models, batch, classes)

        # Count votes
        batch_size = all_preds.shape[1]
        final_preds = []
        agreement_counts = []
        is_confident = []

        for i in range(batch_size):
            votes = all_preds[:, i].cpu().numpy()
            unique, counts = np.unique(votes, return_counts=True)
            max_vote_idx = counts.argmax()
            final_pred = unique[max_vote_idx]
            agreement = counts[max_vote_idx]

            # Check confidence
            avg_conf = all_probs[:, i, final_pred].mean().item()
            confident = agreement >= self.min_agreement and avg_conf >= self.confidence_threshold

            final_preds.append(final_pred)
            agreement_counts.append(agreement)
            is_confident.append(confident)

        return np.array(final_preds), np.array(agreement_counts), np.array(is_confident)

class StackingEnsemble:
    """
    Stacking meta-model: trains meta-learner on base model predictions
    """
    def __init__(self, base_models: Dict, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.is_fitted = False

    def fit(self, X: Dict, y: np.ndarray, meta_model_type: str = "xgboost"):
        """
        Train meta-model on base model predictions
        """
        # Get base model predictions
        base_preds = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X[name])
            else:
                with torch.no_grad():
                    logits = model(X[name])
                    pred = torch.softmax(logits, dim=1).cpu().numpy()
            base_preds.append(pred)

        # Stack predictions
        stacked_features = np.hstack(base_preds)

        # Train meta-model
        if meta_model_type == "xgboost":
            from xgboost import XGBClassifier
            self.meta_model = XGBClassifier(n_estimators=100, max_depth=3)
        elif meta_model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.meta_model = LogisticRegression(max_iter=1000)

        self.meta_model.fit(stacked_features, y)
        self.is_fitted = True

    def predict(self, X: Dict) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Meta-model not fitted")

        base_preds = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X[name])
            else:
                with torch.no_grad():
                    logits = model(X[name])
                    pred = torch.softmax(logits, dim=1).cpu().numpy()
            base_preds.append(pred)

        stacked_features = np.hstack(base_preds)
        return self.meta_model.predict(stacked_features)
