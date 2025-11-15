"""
Ensemble detector with explainability.

Combines multiple detectors for robust prediction with confidence scores
and basic explanations.

⚠️  SECURITY & ETHICS NOTICE ⚠️
Ensemble models are for DEFENSIVE deepfake detection purposes only.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleDetector(nn.Module):
    """
    Ensemble of multiple detectors with voting mechanisms.
    """

    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        voting: str = 'soft'
    ):
        """
        Initialize ensemble detector.

        Args:
            models: List of detector models
            weights: Optional weights for each model (for weighted voting)
            voting: Voting method ('soft' or 'hard')
        """
        super(EnsembleDetector, self).__init__()

        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        if weights is None:
            self.weights = [1.0 / self.num_models] * self.num_models
        else:
            assert len(weights) == self.num_models, "Weights must match number of models"
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]

        self.voting = voting

        logger.info(f"Ensemble initialized with {self.num_models} models")
        logger.info(f"Voting method: {voting}")
        logger.info(f"Weights: {self.weights}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            x: Input tensor

        Returns:
            Ensemble prediction (logits or probabilities)
        """
        outputs = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(output)

        if self.voting == 'soft':
            # Soft voting: average probabilities
            probs = [torch.softmax(out, dim=1) for out in outputs]

            weighted_probs = torch.zeros_like(probs[0])
            for i, prob in enumerate(probs):
                weighted_probs += self.weights[i] * prob

            return weighted_probs

        elif self.voting == 'hard':
            # Hard voting: majority vote
            preds = [out.argmax(dim=1) for out in outputs]
            preds_stacked = torch.stack(preds, dim=1)

            # Get mode (most common prediction)
            mode_preds, _ = torch.mode(preds_stacked, dim=1)

            # Convert to one-hot
            batch_size = x.size(0)
            num_classes = outputs[0].size(1)
            one_hot = torch.zeros(batch_size, num_classes, device=x.device)
            one_hot.scatter_(1, mode_preds.unsqueeze(1), 1)

            return one_hot

        else:
            raise ValueError(f"Unknown voting method: {self.voting}")

    def predict_with_explanation(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict:
        """
        Predict with explanation and confidence scores.

        Args:
            x: Input tensor (single sample or batch)
            threshold: Decision threshold

        Returns:
            Dictionary with predictions, confidence, and explanations
        """
        # Get predictions from each model
        individual_preds = []
        individual_probs = []

        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                output = model(x)
                prob = torch.softmax(output, dim=1)
                pred = (prob[:, 1] > threshold).long()

                individual_preds.append(pred.cpu().numpy())
                individual_probs.append(prob.cpu().numpy())

        # Ensemble prediction
        ensemble_output = self.forward(x)

        if self.voting == 'soft':
            ensemble_pred = (ensemble_output[:, 1] > threshold).long()
            ensemble_conf = ensemble_output[:, 1]
        else:
            ensemble_pred = ensemble_output.argmax(dim=1)
            ensemble_conf = ensemble_output.max(dim=1)[0]

        # Generate explanation
        explanation = self._generate_explanation(
            individual_preds,
            individual_probs,
            ensemble_pred.cpu().numpy(),
            ensemble_conf.cpu().numpy()
        )

        result = {
            'prediction': ensemble_pred.cpu().numpy(),
            'confidence': ensemble_conf.cpu().numpy(),
            'individual_predictions': individual_preds,
            'individual_probabilities': individual_probs,
            'explanation': explanation
        }

        return result

    def _generate_explanation(
        self,
        individual_preds: List[np.ndarray],
        individual_probs: List[np.ndarray],
        ensemble_pred: np.ndarray,
        ensemble_conf: np.ndarray
    ) -> List[str]:
        """
        Generate human-readable explanations for predictions.

        Args:
            individual_preds: Predictions from each model
            individual_probs: Probabilities from each model
            ensemble_pred: Ensemble prediction
            ensemble_conf: Ensemble confidence

        Returns:
            List of explanation strings
        """
        explanations = []

        for i in range(len(ensemble_pred)):
            pred = ensemble_pred[i]
            conf = ensemble_conf[i]

            # Count agreements
            votes_fake = sum([p[i] == 0 for p in individual_preds])
            votes_real = sum([p[i] == 1 for p in individual_preds])

            label = "REAL" if pred == 1 else "FAKE"

            explanation = f"Prediction: {label} (Confidence: {conf:.2%})\n"
            explanation += f"Votes: {votes_real}/{self.num_models} models predict REAL, "
            explanation += f"{votes_fake}/{self.num_models} predict FAKE\n"

            # Confidence interpretation
            if conf >= 0.9:
                explanation += "Very high confidence - strong consensus among models"
            elif conf >= 0.7:
                explanation += "High confidence - good agreement among models"
            elif conf >= 0.5:
                explanation += "Moderate confidence - majority agreement"
            else:
                explanation += "Low confidence - models disagree significantly"

            explanations.append(explanation)

        return explanations


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that adjusts weights based on sample characteristics.
    """

    def __init__(
        self,
        models: List[nn.Module],
        feature_extractor: Optional[nn.Module] = None
    ):
        """
        Initialize adaptive ensemble.

        Args:
            models: List of detector models
            feature_extractor: Feature extractor for weight prediction
        """
        super(AdaptiveEnsemble, self).__init__()

        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        # Weight prediction network
        if feature_extractor is None:
            self.weight_predictor = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_models),
                nn.Softmax(dim=1)
            )
        else:
            self.weight_predictor = feature_extractor

        logger.info(f"Adaptive ensemble initialized with {self.num_models} models")

    def forward(self, x: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with adaptive weighting.

        Args:
            x: Input tensor
            features: Optional pre-computed features for weight prediction

        Returns:
            Ensemble prediction
        """
        # Get predictions from all models
        outputs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                outputs.append(torch.softmax(output, dim=1))

        # Stack outputs
        stacked_outputs = torch.stack(outputs, dim=1)  # (batch, num_models, num_classes)

        # Predict weights (simplified - using dummy features)
        if features is None:
            # Use average of model outputs as features (placeholder)
            features = torch.mean(torch.stack([o[:, 1] for o in outputs], dim=1), dim=1, keepdim=True)
            features = features.expand(-1, 512)  # Dummy expansion

        weights = self.weight_predictor(features)  # (batch, num_models)
        weights = weights.unsqueeze(2)  # (batch, num_models, 1)

        # Weighted average
        ensemble_output = torch.sum(stacked_outputs * weights, dim=1)

        return ensemble_output


def create_ensemble_from_checkpoints(
    checkpoint_paths: List[str],
    model_classes: List[type],
    model_kwargs: List[Dict],
    device: str = 'cpu'
) -> EnsembleDetector:
    """
    Create ensemble from saved checkpoints.

    Args:
        checkpoint_paths: Paths to model checkpoints
        model_classes: Model class types
        model_kwargs: Model initialization arguments
        device: Device to load models on

    Returns:
        Ensemble detector
    """
    models = []

    for i, (path, model_class, kwargs) in enumerate(zip(checkpoint_paths, model_classes, model_kwargs)):
        # Initialize model
        model = model_class(**kwargs)

        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        models.append(model)

        logger.info(f"Loaded model {i+1}/{len(checkpoint_paths)} from {path}")

    ensemble = EnsembleDetector(models)

    return ensemble
