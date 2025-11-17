"""
DetectVoice - Preprocessing Module
Advanced audio feature extraction and augmentation for deepfake detection.
"""

from .features import (
    AdvancedFeatureExtractor,
    extract_mfcc,
    extract_melspectrogram,
    extract_spectrogram,
    extract_chroma,
    extract_cqt,
    extract_zero_crossing_rate,
    extract_spectral_features,
    extract_pitch_formants,
)

from .augmentation import (
    AudioAugmentor,
    SpecAugment,
    add_noise,
    change_gain,
    time_stretch,
    pitch_shift,
)

from .normalization import (
    normalize_audio,
    normalize_features,
    standardize_features,
)

__all__ = [
    # Feature extraction
    "AdvancedFeatureExtractor",
    "extract_mfcc",
    "extract_melspectrogram",
    "extract_spectrogram",
    "extract_chroma",
    "extract_cqt",
    "extract_zero_crossing_rate",
    "extract_spectral_features",
    "extract_pitch_formants",
    # Augmentation
    "AudioAugmentor",
    "SpecAugment",
    "add_noise",
    "change_gain",
    "time_stretch",
    "pitch_shift",
    # Normalization
    "normalize_audio",
    "normalize_features",
    "standardize_features",
]
