"""
Transformer-based Models for Audio Deepfake Detection

Implements state-of-the-art transformer architectures:
- Wav2Vec2 (Facebook/Meta AI)
- HuBERT (Hidden Unit BERT)
- AST (Audio Spectrogram Transformer)

All models support:
- Fine-tuning with layer freezing
- Partial vs. full fine-tuning
- Hybrid training (embeddings + classifier)
"""

from .wav2vec2_detector import Wav2Vec2Detector
from .hubert_detector import HuBERTDetector
from .ast_detector import ASTDetector

__all__ = [
    "Wav2Vec2Detector",
    "HuBERTDetector",
    "ASTDetector",
]
