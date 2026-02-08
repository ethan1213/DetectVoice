"""
Normalization and Standardization Utilities for Audio Features

Implements various normalization techniques for audio and features:
- Audio waveform normalization
- Feature standardization (z-score)
- Min-max normalization
- RMS normalization
- Peak normalization
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional


def normalize_audio(
    waveform: Union[np.ndarray, torch.Tensor],
    method: str = "peak",
    target_level: float = -20.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize audio waveform.

    Args:
        waveform: Input audio waveform
        method: Normalization method ('peak', 'rms', 'lufs')
        target_level: Target level in dB (for RMS/LUFS normalization)

    Returns:
        Normalized waveform
    """
    is_tensor = isinstance(waveform, torch.Tensor)

    if method == "peak":
        # Peak normalization to [-1, 1]
        if is_tensor:
            max_val = torch.abs(waveform).max()
            if max_val > 0:
                normalized = waveform / max_val
            else:
                normalized = waveform
        else:
            max_val = np.abs(waveform).max()
            if max_val > 0:
                normalized = waveform / max_val
            else:
                normalized = waveform

    elif method == "rms":
        # RMS normalization to target level
        if is_tensor:
            rms = torch.sqrt(torch.mean(waveform ** 2))
            target_rms = 10 ** (target_level / 20)
            if rms > 0:
                normalized = waveform * (target_rms / rms)
            else:
                normalized = waveform
        else:
            rms = np.sqrt(np.mean(waveform ** 2))
            target_rms = 10 ** (target_level / 20)
            if rms > 0:
                normalized = waveform * (target_rms / rms)
            else:
                normalized = waveform

    else:
        normalized = waveform

    # Clip to prevent overflow
    if is_tensor:
        normalized = torch.clamp(normalized, -1.0, 1.0)
    else:
        normalized = np.clip(normalized, -1.0, 1.0)

    return normalized


def standardize_features(
    features: Union[np.ndarray, torch.Tensor],
    mean: Optional[Union[np.ndarray, torch.Tensor]] = None,
    std: Optional[Union[np.ndarray, torch.Tensor]] = None,
    axis: Union[int, Tuple[int, ...]] = None,
    epsilon: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Standardize features to zero mean and unit variance (z-score normalization).

    Args:
        features: Input features
        mean: Pre-computed mean (if None, computed from features)
        std: Pre-computed std (if None, computed from features)
        axis: Axis or axes along which to compute statistics
        epsilon: Small value to prevent division by zero

    Returns:
        Standardized features
    """
    is_tensor = isinstance(features, torch.Tensor)

    if mean is None:
        if is_tensor:
            mean = torch.mean(features, dim=axis, keepdim=True)
        else:
            mean = np.mean(features, axis=axis, keepdims=True)

    if std is None:
        if is_tensor:
            std = torch.std(features, dim=axis, keepdim=True)
        else:
            std = np.std(features, axis=axis, keepdims=True)

    # Standardize
    if is_tensor:
        standardized = (features - mean) / (std + epsilon)
    else:
        standardized = (features - mean) / (std + epsilon)

    return standardized


def normalize_features(
    features: Union[np.ndarray, torch.Tensor],
    min_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    max_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    axis: Union[int, Tuple[int, ...]] = None,
    epsilon: float = 1e-8
) -> Union[np.ndarray, torch.Tensor]:
    """
    Min-max normalization to specified range.

    Args:
        features: Input features
        min_val: Pre-computed min (if None, computed from features)
        max_val: Pre-computed max (if None, computed from features)
        feature_range: Target range (min, max)
        axis: Axis or axes along which to compute statistics
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized features
    """
    is_tensor = isinstance(features, torch.Tensor)

    if min_val is None:
        if is_tensor:
            min_val = torch.min(features)
        else:
            min_val = np.min(features)

    if max_val is None:
        if is_tensor:
            max_val = torch.max(features)
        else:
            max_val = np.max(features)

    # Normalize to [0, 1]
    if is_tensor:
        normalized = (features - min_val) / (max_val - min_val + epsilon)
    else:
        normalized = (features - min_val) / (max_val - min_val + epsilon)

    # Scale to target range
    target_min, target_max = feature_range
    if is_tensor:
        normalized = normalized * (target_max - target_min) + target_min
    else:
        normalized = normalized * (target_max - target_min) + target_min

    return normalized


def normalize_spectrogram(
    spectrogram: Union[np.ndarray, torch.Tensor],
    method: str = "standardize"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize spectrogram.

    Args:
        spectrogram: Input spectrogram (freq, time) or (batch, freq, time)
        method: Normalization method ('standardize', 'minmax', 'none')

    Returns:
        Normalized spectrogram
    """
    if method == "standardize":
        # Standardize per frequency bin
        if spectrogram.ndim == 2:
            axis = 1  # Time axis
        else:
            axis = 2  # Time axis for batched input

        return standardize_features(spectrogram, axis=axis)

    elif method == "minmax":
        return normalize_features(spectrogram, feature_range=(0.0, 1.0))

    else:
        return spectrogram


class FeatureNormalizer:
    """
    Feature normalizer that can fit on training data and transform test data.

    Similar to sklearn's StandardScaler but works with both numpy and torch.
    """

    def __init__(self, method: str = "standardize"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('standardize', 'minmax')
        """
        self.method = method
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        self.fitted = False

    def fit(
        self,
        features: Union[np.ndarray, torch.Tensor],
        axis: Union[int, Tuple[int, ...]] = None
    ):
        """
        Compute normalization parameters from training data.

        Args:
            features: Training features
            axis: Axis along which to compute statistics
        """
        is_tensor = isinstance(features, torch.Tensor)

        if self.method == "standardize":
            if is_tensor:
                self.mean = torch.mean(features, dim=axis, keepdim=True)
                self.std = torch.std(features, dim=axis, keepdim=True)
            else:
                self.mean = np.mean(features, axis=axis, keepdims=True)
                self.std = np.std(features, axis=axis, keepdims=True)

        elif self.method == "minmax":
            if is_tensor:
                self.min_val = torch.min(features)
                self.max_val = torch.max(features)
            else:
                self.min_val = np.min(features)
                self.max_val = np.max(features)

        self.fitted = True

    def transform(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Transform features using fitted parameters.

        Args:
            features: Features to transform

        Returns:
            Normalized features
        """
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform")

        if self.method == "standardize":
            return standardize_features(features, self.mean, self.std)
        elif self.method == "minmax":
            return normalize_features(features, self.min_val, self.max_val)
        else:
            return features

    def fit_transform(
        self,
        features: Union[np.ndarray, torch.Tensor],
        axis: Union[int, Tuple[int, ...]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Fit and transform in one step.

        Args:
            features: Features to fit and transform
            axis: Axis along which to compute statistics

        Returns:
            Normalized features
        """
        self.fit(features, axis)
        return self.transform(features)


def compute_global_stats(
    dataset,
    feature_type: str = "melspectrogram"
) -> Tuple[float, float]:
    """
    Compute global mean and std across entire dataset.

    Useful for consistent normalization across train/val/test splits.

    Args:
        dataset: PyTorch dataset or iterable of features
        feature_type: Type of feature being computed

    Returns:
        (global_mean, global_std)
    """
    all_features = []

    for item in dataset:
        if isinstance(item, tuple):
            features = item[0]  # Assuming (features, label) tuple
        else:
            features = item

        if isinstance(features, torch.Tensor):
            features = features.numpy()

        all_features.append(features.flatten())

    all_features = np.concatenate(all_features)

    global_mean = np.mean(all_features)
    global_std = np.std(all_features)

    return global_mean, global_std
