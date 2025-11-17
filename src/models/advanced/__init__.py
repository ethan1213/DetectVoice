"""
Advanced Specialized Models for Audio Deepfake Detection
ECAPA-TDNN, ResNet-Audio, QuartzNet, Conformer, Harmonic CNN
"""

from .ecapa_tdnn import ECAPATDNNDetector
from .resnet_audio import ResNetAudioDetector
from .quartznet import QuartzNetDetector
from .conformer import ConformerDetector
from .harmonic_cnn import HarmonicCNN

__all__ = [
    "ECAPATDNNDetector",
    "ResNetAudioDetector",
    "QuartzNetDetector",
    "ConformerDetector",
    "HarmonicCNN",
]
