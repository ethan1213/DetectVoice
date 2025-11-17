"""
Classical Machine Learning Models for Audio Deepfake Detection

Implements traditional ML approaches:
- SVM (Support Vector Machine)
- XGBoost
- Random Forest
- Logistic Regression

These models work with hand-crafted features (MFCC, spectral features, etc.)
and can be very effective for deepfake detection.
"""

from .svm_detector import SVMDetector
from .xgboost_detector import XGBoostDetector
from .random_forest_detector import RandomForestDetector
from .logistic_regression_detector import LogisticRegressionDetector
from .classical_ensemble import ClassicalEnsemble

__all__ = [
    "SVMDetector",
    "XGBoostDetector",
    "RandomForestDetector",
    "LogisticRegressionDetector",
    "ClassicalEnsemble",
]
