"""
Logistic Regression Detector for Audio Deepfake Detection

Simple, fast, and interpretable baseline model.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import Optional, Dict
import warnings


class LogisticRegressionDetector:
    """
    Logistic Regression-based audio deepfake detector.

    Fast, interpretable baseline. Good for high-dimensional features.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        """
        Initialize Logistic Regression detector.

        Args:
            C: Inverse regularization strength
            penalty: Regularization ('l1', 'l2', 'elasticnet', 'none')
            solver: Optimization algorithm
            max_iter: Maximum iterations
            class_weight: Class weights ('balanced' or None)
            random_state: Random seed
        """
        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        # Create pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(
                C=C,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=-1,
            ))
        ])

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Logistic Regression."""
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return self.model.score(X, y)

    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients (feature weights)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        return self.model.named_steps['logreg'].coef_

    def save(self, path: str):
        """Save model."""
        if not self.is_fitted:
            warnings.warn("Saving unfitted model")

        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model."""
        self.model = joblib.load(path)
        self.is_fitted = True

    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'C': self.C,
            'penalty': self.penalty,
            'solver': self.solver,
            'max_iter': self.max_iter,
            'class_weight': self.class_weight,
        }
