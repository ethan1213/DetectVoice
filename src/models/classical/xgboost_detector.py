"""
XGBoost Detector for Audio Deepfake Detection

XGBoost (Extreme Gradient Boosting) is highly effective for audio
classification tasks with hand-crafted features.
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Optional, Dict
import warnings


class XGBoostDetector:
    """
    XGBoost-based audio deepfake detector.

    Uses gradient boosting on hand-crafted audio features.
    Often achieves SOTA results with classical features.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        """
        Initialize XGBoost detector.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            gamma: Minimum loss reduction for split
            min_child_weight: Minimum sum of instance weight in child
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            scale_pos_weight: Balance of positive/negative weights
            random_state: Random seed
            use_gpu: Use GPU acceleration
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.use_gpu = use_gpu

        # Initialize scaler
        self.scaler = StandardScaler()

        # Initialize XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            tree_method='gpu_hist' if use_gpu else 'hist',
            eval_metric='logloss',
            use_label_encoder=False,
        )

        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 10,
        verbose: bool = True
    ):
        """
        Train XGBoost model.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            eval_set: Validation set [(X_val, y_val)]
            early_stopping_rounds: Early stopping patience
            verbose: Print training progress
        """
        # Flatten if needed
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Scale eval set if provided
        if eval_set is not None:
            eval_set_scaled = []
            for X_eval, y_eval in eval_set:
                if X_eval.ndim > 2:
                    X_eval = X_eval.reshape(X_eval.shape[0], -1)
                X_eval_scaled = self.scaler.transform(X_eval)
                eval_set_scaled.append((X_eval_scaled, y_eval))
        else:
            eval_set_scaled = None

        # Train
        self.model.fit(
            X_scaled,
            y,
            eval_set=eval_set_scaled,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose
        )

        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importances."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")

        return self.model.feature_importances_

    def save(self, path: str):
        """Save model and scaler."""
        if not self.is_fitted:
            warnings.warn("Saving unfitted model")

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'params': self.get_params()
        }, path)

    def load(self, path: str):
        """Load model and scaler."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_fitted = True

    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
        }
