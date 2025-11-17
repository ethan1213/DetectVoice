"""
Advanced Feature Extraction for Audio Deepfake Detection

Implements comprehensive feature extraction techniques:
- MFCC (Mel-Frequency Cepstral Coefficients)
- Mel Spectrogram
- STFT Spectrogram (log-scaled)
- Chroma STFT
- Constant-Q Transform (CQT)
- Spectral features (centroid, bandwidth, rolloff, contrast)
- Zero Crossing Rate
- Pitch and Formant tracking
- Raw waveform features
"""

import numpy as np
import librosa
import torch
import torchaudio
from typing import Optional, Tuple, Dict, Union
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """
    Comprehensive feature extractor for audio deepfake detection.

    Supports multiple feature types and can extract them simultaneously
    for multi-view learning approaches.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
        n_chroma: int = 12,
        cqt_bins: int = 84,
    ):
        """
        Initialize feature extractor.

        Args:
            sample_rate: Audio sample rate in Hz
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            n_mels: Number of Mel bands
            n_mfcc: Number of MFCCs to extract
            fmin: Minimum frequency
            fmax: Maximum frequency
            n_chroma: Number of chroma bins
            cqt_bins: Number of CQT bins per octave
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.n_chroma = n_chroma
        self.cqt_bins = cqt_bins

        # Initialize torchaudio transforms for GPU acceleration
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
        )

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'f_min': fmin,
                'f_max': self.fmax,
            }
        )

    def extract_all_features(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        features: list = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract multiple features at once.

        Args:
            waveform: Audio waveform (1D array or tensor)
            features: List of feature names to extract. If None, extracts all.
                     Options: ['mfcc', 'melspec', 'spec', 'chroma', 'cqt',
                              'spectral', 'zcr', 'pitch']

        Returns:
            Dictionary mapping feature names to feature arrays
        """
        if features is None:
            features = ['mfcc', 'melspec', 'spec', 'chroma', 'spectral']

        # Convert to numpy if tensor
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform

        feature_dict = {}

        if 'mfcc' in features:
            feature_dict['mfcc'] = self.extract_mfcc(waveform)

        if 'melspec' in features:
            feature_dict['melspec'] = self.extract_melspectrogram(waveform)

        if 'spec' in features:
            feature_dict['spec'] = self.extract_spectrogram(waveform_np)

        if 'chroma' in features:
            feature_dict['chroma'] = self.extract_chroma(waveform_np)

        if 'cqt' in features:
            feature_dict['cqt'] = self.extract_cqt(waveform_np)

        if 'spectral' in features:
            spectral_feats = self.extract_spectral_features(waveform_np)
            feature_dict.update(spectral_feats)

        if 'zcr' in features:
            feature_dict['zcr'] = self.extract_zcr(waveform_np)

        if 'pitch' in features:
            pitch_feats = self.extract_pitch_formants(waveform_np)
            feature_dict.update(pitch_feats)

        return feature_dict

    def extract_mfcc(self, waveform: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Extract MFCC features."""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        mfcc = self.mfcc_transform(waveform)
        return mfcc.squeeze(0).numpy()

    def extract_melspectrogram(
        self,
        waveform: Union[np.ndarray, torch.Tensor],
        to_db: bool = True
    ) -> np.ndarray:
        """Extract Mel spectrogram."""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        melspec = self.mel_transform(waveform)

        if to_db:
            melspec = torchaudio.transforms.AmplitudeToDB()(melspec)

        return melspec.squeeze(0).numpy()

    def extract_spectrogram(
        self,
        waveform: np.ndarray,
        log_scale: bool = True
    ) -> np.ndarray:
        """Extract STFT spectrogram (log-scaled)."""
        D = librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        magnitude = np.abs(D)

        if log_scale:
            magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

        return magnitude

    def extract_chroma(self, waveform: np.ndarray) -> np.ndarray:
        """Extract Chroma STFT features."""
        chroma = librosa.feature.chroma_stft(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        return chroma

    def extract_cqt(self, waveform: np.ndarray) -> np.ndarray:
        """Extract Constant-Q Transform."""
        cqt = librosa.cqt(
            y=waveform,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.cqt_bins,
            bins_per_octave=12
        )

        # Convert to dB scale
        cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
        return cqt_db

    def extract_spectral_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive spectral features.

        Returns dictionary with:
        - spectral_centroid
        - spectral_bandwidth
        - spectral_rolloff
        - spectral_contrast
        """
        features = {}

        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]

        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]

        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )[0]

        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        return features

    def extract_zcr(self, waveform: np.ndarray) -> np.ndarray:
        """Extract Zero Crossing Rate."""
        zcr = librosa.feature.zero_crossing_rate(
            waveform,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )[0]
        return zcr

    def extract_pitch_formants(
        self,
        waveform: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract pitch (F0) and formant features.

        Uses librosa's piptrack for pitch detection.
        Note: For advanced formant analysis, consider using Praat via parselmouth.
        """
        features = {}

        # Pitch tracking (F0)
        pitches, magnitudes = librosa.piptrack(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=75,  # Typical male voice
            fmax=600  # Typical female voice
        )

        # Extract pitch contour
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch)

        features['pitch'] = np.array(pitch_contour)

        return features

    def extract_raw_waveform(
        self,
        waveform: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Extract raw waveform features (for 1D CNNs).

        Args:
            waveform: Raw audio waveform
            normalize: Whether to normalize to [-1, 1]

        Returns:
            Normalized waveform if normalize=True
        """
        if normalize:
            # Normalize to [-1, 1]
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / max_val

        return waveform


# ============================================================================
# Standalone utility functions
# ============================================================================

def extract_mfcc(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """
    Extract MFCC features (standalone function).

    Args:
        waveform: Audio waveform
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCCs
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        MFCC features (n_mfcc, time_steps)
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc


def extract_melspectrogram(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    to_db: bool = True
) -> np.ndarray:
    """
    Extract Mel spectrogram (standalone function).

    Args:
        waveform: Audio waveform
        sample_rate: Sample rate in Hz
        n_mels: Number of Mel bands
        n_fft: FFT window size
        hop_length: Hop length
        to_db: Convert to dB scale

    Returns:
        Mel spectrogram (n_mels, time_steps)
    """
    melspec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    if to_db:
        melspec = librosa.power_to_db(melspec, ref=np.max)

    return melspec


def extract_spectrogram(
    waveform: np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
    log_scale: bool = True
) -> np.ndarray:
    """
    Extract STFT spectrogram (standalone function).

    Args:
        waveform: Audio waveform
        n_fft: FFT window size
        hop_length: Hop length
        log_scale: Apply log scaling

    Returns:
        Spectrogram (freq_bins, time_steps)
    """
    D = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)

    if log_scale:
        magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

    return magnitude


def extract_chroma(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Extract Chroma STFT features (standalone function)."""
    chroma = librosa.feature.chroma_stft(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return chroma


def extract_cqt(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 512,
    n_bins: int = 84
) -> np.ndarray:
    """Extract Constant-Q Transform (standalone function)."""
    cqt = librosa.cqt(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins
    )
    return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)


def extract_zero_crossing_rate(
    waveform: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    """Extract Zero Crossing Rate (standalone function)."""
    return librosa.feature.zero_crossing_rate(
        waveform,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]


def extract_spectral_features(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive spectral features (standalone function).

    Returns:
        Dictionary with spectral_centroid, spectral_bandwidth,
        spectral_rolloff, and spectral_contrast
    """
    features = {}

    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]

    features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]

    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )[0]

    features['spectral_contrast'] = librosa.feature.spectral_contrast(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
    )

    return features


def extract_pitch_formants(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, np.ndarray]:
    """
    Extract pitch (F0) features (standalone function).

    Note: For advanced formant analysis, install parselmouth:
    pip install praat-parselmouth

    Returns:
        Dictionary with 'pitch' contour
    """
    pitches, magnitudes = librosa.piptrack(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=75,
        fmax=600
    )

    pitch_contour = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_contour.append(pitch)

    return {'pitch': np.array(pitch_contour)}
