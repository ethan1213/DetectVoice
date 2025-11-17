"""
Classical Model Ensemble

Combines SVM, XGBoost, Random Forest, and Logistic Regression
for improved performance.
"""

import numpy as np
from typing import List, Dict, Optional
import joblib
import warnings

from .svm_detector import SVMDetector
from .xgboost_detector import XGBoostDetector
from .random_forest_detector import RandomForestDetector
from .logistic_regression_detector import LogisticRegressionDetector


class ClassicalEnsemble:
    """
    Ensemble of classical ML models.

    Combines multiple models using voting or stacking.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        voting: str = "soft",  # 'soft' or 'hard'
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize classical ensemble.

        Args:
            models: List of model names to include
                   ['svm', 'xgboost', 'random_forest', 'logistic_regression']
            voting: Voting strategy ('soft' = avg probabilities, 'hard' = majority vote)
            weights: Model weights for soft voting
        """
        if models is None:
            models = ['svm', 'xgboost', 'random_forest', 'logistic_regression']

        self.model_names = models
        self.voting = voting
        self.weights = weights
        self.models = {}
        self.is_fitted = False

        # Initialize models
        if 'svm' in models:
            self.models['svm'] = SVMDetector()

        if 'xgboost' in models:
            self.models['xgboost'] = XGBoostDetector()

        if 'random_forest' in models:
            self.models['random_forest'] = RandomForestDetector()

        if 'logistic_regression' in models:
            self.models['logistic_regression'] = LogisticRegressionDetector()

        # Set default weights if not provided
        if self.weights is None:
            self.weights = [1.0 / len(self.models)] * len(self.models)

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params):
        """
        Train all models in ensemble.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            **fit_params: Additional parameters for specific models
        """
        print(f"Training ensemble with {len(self.models)} models...")

        for i, (name, model) in enumerate(self.models.items()):
            print(f"Training {name} ({i+1}/{len(self.models)})...")

            # Extract model-specific fit params
            model_fit_params = fit_params.get(name, {})

            # Train model
            if name == 'xgboost' and 'eval_set' in model_fit_params:
                model.fit(X, y, **model_fit_params)
            else:
                model.fit(X, y)

        self.is_fitted = True
        print("Ensemble training complete!")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using ensemble.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        if self.voting == "soft":
            # Soft voting: average probabilities
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)

        else:
            # Hard voting: majority vote
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred)

            predictions = np.array(predictions)  # (n_models, n_samples)

            # Weighted majority vote
            weighted_votes = np.zeros((X.shape[0], 2))
            for i, (name, model) in enumerate(self.models.items()):
                for j in range(X.shape[0]):
                    weighted_votes[j, predictions[i, j]] += self.weights[i]

            return np.argmax(weighted_votes, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using ensemble.

        Args:
            X: Features

        Returns:
            Probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble must be fitted before prediction")

        # Collect probabilities from all models
        all_proba = []
        for name, model in self.models.items():
            proba = model.predict_proba(X)
            all_proba.append(proba)

        all_proba = np.array(all_proba)  # (n_models, n_samples, n_classes)

        # Weighted average
        weights = np.array(self.weights).reshape(-1, 1, 1)
        ensemble_proba = np.sum(all_proba * weights, axis=0)

        # Normalize to ensure probabilities sum to 1
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)

        return ensemble_proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_individual_scores(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Get accuracy scores for each individual model."""
        scores = {}
        for name, model in self.models.items():
            scores[name] = model.score(X, y)

        scores['ensemble'] = self.score(X, y)

        return scores

    def save(self, path: str):
        """Save ensemble."""
        if not self.is_fitted:
            warnings.warn("Saving unfitted ensemble")

        joblib.dump({
            'models': self.models,
            'model_names': self.model_names,
            'voting': self.voting,
            'weights': self.weights,
        }, path)

    def load(self, path: str):
        """Load ensemble."""
        data = joblib.load(path)
        self.models = data['models']
        self.model_names = data['model_names']
        self.voting = data['voting']
        self.weights = data['weights']
        self.is_fitted = True
