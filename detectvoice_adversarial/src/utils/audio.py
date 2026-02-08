"""
Audio utilities: loading, feature extraction, and augmentation.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This code is part of a research project for audio deepfake detection and adversarial robustness.
Use responsibly and in compliance with local laws and ethical guidelines.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from .logger import get_logger

logger = get_logger(__name__)


class AudioFeatureExtractor:
    """Extract various audio features for deep learning models."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 128,
        n_mfcc: int = 40,
        power: float = 2.0
    ):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCCs
            power: Exponent for magnitude spectrogram
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        # Transforms
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=power
        )

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power
        )

        self.mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels
            }
        )

        self.amplitude_to_db = T.AmplitudeToDB()

    def extract_mel_spectrogram(
        self,
        waveform: torch.Tensor,
        to_db: bool = True
    ) -> torch.Tensor:
        """
        Extract mel-spectrogram from waveform.

        Args:
            waveform: Input waveform (channels, time)
            to_db: Convert to decibel scale

        Returns:
            Mel-spectrogram (channels, n_mels, time)
        """
        mel_spec = self.mel_spectrogram(waveform)
        if to_db:
            mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec

    def extract_spectrogram(
        self,
        waveform: torch.Tensor,
        to_db: bool = True
    ) -> torch.Tensor:
        """Extract spectrogram from waveform."""
        spec = self.spectrogram(waveform)
        if to_db:
            spec = self.amplitude_to_db(spec)
        return spec

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MFCCs from waveform."""
        return self.mfcc(waveform)

    def normalize(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize features to zero mean and unit variance."""
        mean = features.mean()
        std = features.std()
        return (features - mean) / (std + eps)


def load_audio(
    audio_path: Path,
    sample_rate: int = 16000,
    duration: Optional[float] = None,
    offset: float = 0.0
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if needed.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        duration: Maximum duration in seconds (None for full audio)
        offset: Start offset in seconds

    Returns:
        Tuple of (waveform, sample_rate)
    """
    audio_path = Path(audio_path)

    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Apply offset
    if offset > 0:
        offset_samples = int(offset * sr)
        waveform = waveform[:, offset_samples:]

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


def save_audio(
    waveform: torch.Tensor,
    audio_path: Path,
    sample_rate: int = 16000
) -> None:
    """
    Save waveform to audio file.

    Args:
        waveform: Waveform tensor
        audio_path: Path to save audio
        sample_rate: Sample rate
    """
    audio_path = Path(audio_path)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(audio_path), waveform, sample_rate)
    logger.debug(f"Audio saved to: {audio_path}")


def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio in dB.

    Args:
        clean: Clean signal
        noisy: Noisy signal

    Returns:
        SNR in dB
    """
    noise = noisy - clean
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()


class AudioAugmentation:
    """Audio data augmentation for training robustness."""

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_prob: float = 0.3,
        noise_snr_db: Tuple[float, float] = (10.0, 30.0),
        gain_range: Tuple[float, float] = (0.8, 1.2)
    ):
        """
        Initialize audio augmentation.

        Args:
            sample_rate: Sample rate
            noise_prob: Probability of adding noise
            noise_snr_db: SNR range for noise injection (min, max) in dB
            gain_range: Range for volume augmentation
        """
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.noise_snr_db = noise_snr_db
        self.gain_range = gain_range

    def add_noise(self, waveform: torch.Tensor, snr_db: Optional[float] = None) -> torch.Tensor:
        """
        Add white noise to waveform with specified SNR.

        Args:
            waveform: Input waveform
            snr_db: Target SNR in dB (random if None)

        Returns:
            Noisy waveform
        """
        if np.random.random() > self.noise_prob:
            return waveform

        if snr_db is None:
            snr_db = np.random.uniform(self.noise_snr_db[0], self.noise_snr_db[1])

        # Generate white noise
        noise = torch.randn_like(waveform)

        # Compute signal and noise power
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        # Scale noise to achieve target SNR
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise = noise * torch.sqrt(target_noise_power / noise_power)

        return waveform + noise

    def apply_gain(self, waveform: torch.Tensor, gain: Optional[float] = None) -> torch.Tensor:
        """Apply random gain/volume change."""
        if gain is None:
            gain = np.random.uniform(self.gain_range[0], self.gain_range[1])
        return waveform * gain

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations."""
        waveform = self.add_noise(waveform)
        waveform = self.apply_gain(waveform)
        return waveform


class SpecAugment:
    """
    SpecAugment for spectrograms.
    Ref: https://arxiv.org/abs/1904.08779
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ):
        """
        Initialize SpecAugment.

        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks
        """
        self.freq_masking = T.FrequencyMasking(freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def apply(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.

        Args:
            spectrogram: Input spectrogram (batch, freq, time) or (freq, time)

        Returns:
            Augmented spectrogram
        """
        for _ in range(self.n_freq_masks):
            spectrogram = self.freq_masking(spectrogram)

        for _ in range(self.n_time_masks):
            spectrogram = self.time_masking(spectrogram)

        return spectrogram
