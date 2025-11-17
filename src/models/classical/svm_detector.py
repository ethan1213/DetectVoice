"""
SVM (Support Vector Machine) Detector

SVMs are effective for deepfake detection when using proper kernel functions
and features. Works well with MFCC and spectral features.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import Optional, Dict, Tuple
import warnings


class SVMDetector:
    """
    SVM-based audio deepfake detector.

    Uses hand-crafted features (MFCC, spectral features) with
    support vector classification.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        class_weight: Optional[str] = "balanced",
        probability: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize SVM detector.

        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            class_weight: Class weights ('balanced' or None)
            probability: Enable probability estimates
            random_state: Random seed
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.class_weight = class_weight
        self.probability = probability
        self.random_state = random_state

        # Create pipeline with scaler + SVM
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                class_weight=class_weight,
                probability=probability,
                random_state=random_state,
                cache_size=1000,  # Speed up training
            ))
        ])

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train SVM on features.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
        """
        # Flatten features if needed (e.g., from 2D MFCC)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if not self.probability:
            warnings.warn("Probability estimation was not enabled. Returning 0/1 probabilities.")
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Features
            y: True labels

        Returns:
            Accuracy score
        """
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        return self.model.score(X, y)

    def save(self, path: str):
        """Save model to disk."""
        if not self.is_fitted:
            warnings.warn("Saving unfitted model")

        joblib.dump(self.model, path)

    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_fitted = True

    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        return self.model.named_steps['svm'].support_vectors_

    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'class_weight': self.class_weight,
            'probability': self.probability,
            'n_support_vectors': len(self.model.named_steps['svm'].support_) if self.is_fitted else None
        }
