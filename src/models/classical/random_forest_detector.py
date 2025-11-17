"""
Random Forest Detector for Audio Deepfake Detection

Random Forest is robust, interpretable, and effective for audio classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from typing import Optional, Dict
import warnings


class RandomForestDetector:
    """
    Random Forest-based audio deepfake detector.

    Ensemble of decision trees, robust to overfitting and provides
    feature importance rankings.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: Optional[str] = "balanced",
        bootstrap: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Random Forest detector.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth (None = unlimited)
            min_samples_split: Minimum samples to split node
            min_samples_leaf: Minimum samples in leaf
            max_features: Number of features for best split
            class_weight: Class weights ('balanced' or None)
            bootstrap: Use bootstrap samples
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Create pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                bootstrap=bootstrap,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0,
            ))
        ])

        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest."""
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

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        return self.model.named_steps['rf'].feature_importances_

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
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
        }
