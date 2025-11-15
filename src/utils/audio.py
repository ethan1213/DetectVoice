"""
Audio utilities for DetectVoice.
Compatible with existing codebase.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    duration: Optional[float] = None
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if needed.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        duration: Duration in seconds (None for full audio)

    Returns:
        Tuple of (waveform, sample_rate)
    """
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Trim or pad to duration
    if duration is not None:
        target_length = int(sample_rate * duration)
        current_length = waveform.shape[1]

        if current_length > target_length:
            waveform = waveform[:, :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

    return waveform, sample_rate


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 256,
    n_mels: int = 128,
    to_db: bool = True
) -> torch.Tensor:
    """
    Extract mel-spectrogram from waveform.

    Args:
        waveform: Input waveform (channels, time)
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        n_mels: Number of mel bands
        to_db: Convert to decibel scale

    Returns:
        Mel-spectrogram (channels, n_mels, time)
    """
    mel_spec_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_spec = mel_spec_transform(waveform)

    if to_db:
        amplitude_to_db = T.AmplitudeToDB()
        mel_spec = amplitude_to_db(mel_spec)

    return mel_spec


def normalize_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    """
    Normalize spectrogram to zero mean and unit variance.

    Args:
        spectrogram: Input spectrogram

    Returns:
        Normalized spectrogram
    """
    mean = spectrogram.mean()
    std = spectrogram.std()
    return (spectrogram - mean) / (std + 1e-8)
