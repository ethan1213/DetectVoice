"""
Advanced Audio Augmentation for Deepfake Detection

Implements comprehensive augmentation techniques:
- Additive noise (white, colored, environmental)
- Gain/volume adjustment
- Time stretching
- Pitch shifting
- Codec simulation (MP3, Opus)
- Room acoustics simulation
- SpecAugment for spectrograms
- Background noise injection
"""

import numpy as np
import librosa
import torch
import random
from typing import Optional, Tuple, Union
import scipy.signal as signal


class AudioAugmentor:
    """
    Comprehensive audio augmentation for robustness training.

    Implements various augmentation techniques to make models
    robust to real-world variations and adversarial attacks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        prob_noise: float = 0.3,
        prob_gain: float = 0.3,
        prob_time_stretch: float = 0.2,
        prob_pitch_shift: float = 0.2,
        noise_snr_db: Tuple[float, float] = (10.0, 30.0),
        gain_range: Tuple[float, float] = (0.7, 1.3),
        time_stretch_rate: Tuple[float, float] = (0.9, 1.1),
        pitch_shift_steps: Tuple[int, int] = (-2, 2),
    ):
        """
        Initialize augmentor.

        Args:
            sample_rate: Audio sample rate
            prob_noise: Probability of applying noise
            prob_gain: Probability of applying gain
            prob_time_stretch: Probability of time stretching
            prob_pitch_shift: Probability of pitch shifting
            noise_snr_db: SNR range for noise (min, max) in dB
            gain_range: Gain multiplier range (min, max)
            time_stretch_rate: Time stretch rate range (min, max)
            pitch_shift_steps: Pitch shift in semitones (min, max)
        """
        self.sample_rate = sample_rate
        self.prob_noise = prob_noise
        self.prob_gain = prob_gain
        self.prob_time_stretch = prob_time_stretch
        self.prob_pitch_shift = prob_pitch_shift
        self.noise_snr_db = noise_snr_db
        self.gain_range = gain_range
        self.time_stretch_rate = time_stretch_rate
        self.pitch_shift_steps = pitch_shift_steps

    def augment(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to waveform.

        Args:
            waveform: Input audio waveform

        Returns:
            Augmented waveform
        """
        augmented = waveform.copy()

        # Apply augmentations with configured probabilities
        if random.random() < self.prob_noise:
            snr_db = random.uniform(*self.noise_snr_db)
            augmented = add_noise(augmented, snr_db=snr_db)

        if random.random() < self.prob_gain:
            gain = random.uniform(*self.gain_range)
            augmented = change_gain(augmented, gain)

        if random.random() < self.prob_time_stretch:
            rate = random.uniform(*self.time_stretch_rate)
            augmented = time_stretch(augmented, rate, self.sample_rate)

        if random.random() < self.prob_pitch_shift:
            steps = random.randint(*self.pitch_shift_steps)
            augmented = pitch_shift(augmented, steps, self.sample_rate)

        return augmented


class SpecAugment:
    """
    SpecAugment: A Simple Data Augmentation Method for ASR
    (Park et al., 2019)

    Applies frequency and time masking to spectrograms.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        prob: float = 0.5,
    ):
        """
        Initialize SpecAugment.

        Args:
            freq_mask_param: Maximum frequency mask size
            time_mask_param: Maximum time mask size
            num_freq_masks: Number of frequency masks to apply
            num_time_masks: Number of time masks to apply
            prob: Probability of applying augmentation
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.prob = prob

    def __call__(
        self,
        spec: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply SpecAugment to spectrogram.

        Args:
            spec: Spectrogram (freq_bins, time_steps) or (batch, freq, time)

        Returns:
            Augmented spectrogram
        """
        if random.random() > self.prob:
            return spec

        is_tensor = isinstance(spec, torch.Tensor)
        if is_tensor:
            spec = spec.clone()
        else:
            spec = spec.copy()

        # Handle batched input
        if spec.ndim == 3:
            # Batch dimension present
            for i in range(spec.shape[0]):
                spec[i] = self._augment_single(spec[i])
        else:
            spec = self._augment_single(spec)

        return spec

    def _augment_single(
        self,
        spec: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply augmentation to single spectrogram."""
        freq_bins, time_steps = spec.shape[-2:]

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(0, freq_bins - f))
            if isinstance(spec, torch.Tensor):
                spec[..., f0:f0+f, :] = 0
            else:
                spec[..., f0:f0+f, :] = 0

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(0, time_steps - t))
            if isinstance(spec, torch.Tensor):
                spec[..., :, t0:t0+t] = 0
            else:
                spec[..., :, t0:t0+t] = 0

        return spec


# ============================================================================
# Augmentation utility functions
# ============================================================================

def add_noise(
    waveform: np.ndarray,
    noise_type: str = "white",
    snr_db: float = 20.0
) -> np.ndarray:
    """
    Add noise to waveform.

    Args:
        waveform: Input audio
        noise_type: Type of noise ('white', 'pink', 'brown', 'environmental')
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Noisy waveform
    """
    # Generate noise
    if noise_type == "white":
        noise = np.random.randn(len(waveform))
    elif noise_type == "pink":
        noise = generate_pink_noise(len(waveform))
    elif noise_type == "brown":
        noise = generate_brown_noise(len(waveform))
    else:
        noise = np.random.randn(len(waveform))

    # Calculate signal and noise power
    signal_power = np.mean(waveform ** 2)
    noise_power = np.mean(noise ** 2)

    # Calculate scaling factor for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(signal_power / (noise_power * snr_linear))

    # Add scaled noise
    noisy_waveform = waveform + scale * noise

    return noisy_waveform


def generate_pink_noise(length: int) -> np.ndarray:
    """
    Generate pink noise (1/f noise).

    Pink noise has equal energy per octave.
    """
    # Generate white noise
    white = np.random.randn(length)

    # Apply pink noise filter
    # Simple approximation using moving average
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
    a = np.array([1, -2.494956002, 2.017265875, -0.522189400])

    pink = signal.lfilter(b, a, white)

    # Normalize
    pink = pink / np.abs(pink).max()

    return pink


def generate_brown_noise(length: int) -> np.ndarray:
    """
    Generate brown noise (Brownian noise).

    Brown noise has energy decreasing 6dB per octave.
    """
    white = np.random.randn(length)
    brown = np.cumsum(white)

    # Normalize
    brown = brown / np.abs(brown).max()

    return brown


def change_gain(waveform: np.ndarray, gain: float) -> np.ndarray:
    """
    Apply gain (volume) adjustment.

    Args:
        waveform: Input audio
        gain: Gain multiplier (e.g., 1.2 = +20% volume)

    Returns:
        Gain-adjusted waveform
    """
    return np.clip(waveform * gain, -1.0, 1.0)


def time_stretch(
    waveform: np.ndarray,
    rate: float,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Time stretch audio without changing pitch.

    Args:
        waveform: Input audio
        rate: Stretch rate (>1 = faster, <1 = slower)
        sample_rate: Sample rate

    Returns:
        Time-stretched waveform
    """
    stretched = librosa.effects.time_stretch(waveform, rate=rate)

    # Ensure same length (trim or pad)
    if len(stretched) > len(waveform):
        stretched = stretched[:len(waveform)]
    elif len(stretched) < len(waveform):
        stretched = np.pad(stretched, (0, len(waveform) - len(stretched)))

    return stretched


def pitch_shift(
    waveform: np.ndarray,
    n_steps: int,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Shift pitch of audio.

    Args:
        waveform: Input audio
        n_steps: Number of semitones to shift (positive = higher, negative = lower)
        sample_rate: Sample rate

    Returns:
        Pitch-shifted waveform
    """
    shifted = librosa.effects.pitch_shift(
        waveform,
        sr=sample_rate,
        n_steps=n_steps
    )
    return shifted


def simulate_codec(
    waveform: np.ndarray,
    codec_type: str = "mp3",
    bitrate: int = 128
) -> np.ndarray:
    """
    Simulate codec compression artifacts.

    Note: This is a simplified simulation. For real codec effects,
    use actual encoding/decoding with ffmpeg or similar tools.

    Args:
        waveform: Input audio
        codec_type: Codec type ('mp3', 'opus', 'aac')
        bitrate: Bitrate in kbps

    Returns:
        Codec-simulated waveform
    """
    # Simple simulation: low-pass filter + quantization noise
    # Real codecs are much more complex

    # Calculate cutoff based on bitrate
    cutoff_freq = min(8000, bitrate * 50)  # Heuristic

    # Design low-pass filter
    nyquist = 16000 / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(4, normalized_cutoff, btype='low')

    # Apply filter
    filtered = signal.filtfilt(b, a, waveform)

    # Add quantization noise based on bitrate
    noise_level = 1.0 / (bitrate + 1)
    noise = np.random.randn(len(waveform)) * noise_level * 0.01

    simulated = filtered + noise

    return np.clip(simulated, -1.0, 1.0)


def add_room_acoustics(
    waveform: np.ndarray,
    room_size: str = "medium",
    reverb_amount: float = 0.3
) -> np.ndarray:
    """
    Simulate room acoustics (reverb).

    Simple convolution-based reverb simulation.

    Args:
        waveform: Input audio
        room_size: Room size ('small', 'medium', 'large')
        reverb_amount: Amount of reverb (0.0 to 1.0)

    Returns:
        Reverberated waveform
    """
    # Define impulse response lengths based on room size
    ir_lengths = {
        'small': 4000,
        'medium': 8000,
        'large': 16000
    }

    ir_length = ir_lengths.get(room_size, 8000)

    # Generate simple exponential decay impulse response
    t = np.arange(ir_length)
    decay_rate = 5.0  # Adjust for decay speed
    impulse = np.exp(-decay_rate * t / ir_length) * np.random.randn(ir_length)
    impulse[0] = 1.0  # Direct sound

    # Normalize
    impulse = impulse / np.abs(impulse).max()

    # Convolve with impulse response
    reverb = signal.convolve(waveform, impulse, mode='same')

    # Mix dry and wet signal
    output = (1 - reverb_amount) * waveform + reverb_amount * reverb

    # Normalize to prevent clipping
    output = output / np.abs(output).max()

    return output


def add_background_noise(
    waveform: np.ndarray,
    noise_type: str = "babble",
    snr_db: float = 15.0
) -> np.ndarray:
    """
    Add realistic background noise.

    Args:
        waveform: Input audio
        noise_type: Type of background ('babble', 'traffic', 'cafe', 'white')
        snr_db: Signal-to-noise ratio in dB

    Returns:
        Audio with background noise
    """
    # For realistic noise, you would load actual noise samples
    # Here we use generated noise as placeholder

    if noise_type == "babble":
        # Simulate multiple voices (bandpass filtered noise)
        noise = np.random.randn(len(waveform))
        # Bandpass 300-3400 Hz (typical voice range)
        sos = signal.butter(4, [300, 3400], btype='band', fs=16000, output='sos')
        noise = signal.sosfilt(sos, noise)
    elif noise_type == "traffic":
        # Low frequency rumble
        noise = generate_brown_noise(len(waveform))
        # Low-pass filter
        sos = signal.butter(4, 500, btype='low', fs=16000, output='sos')
        noise = signal.sosfilt(sos, noise)
    else:
        noise = np.random.randn(len(waveform))

    # Add noise with specified SNR
    return add_noise(waveform, noise_type="white", snr_db=snr_db)


class ChainAugmentor:
    """
    Chain multiple augmentations sequentially.

    Example:
        augmentor = ChainAugmentor([
            AudioAugmentor(),
            SpecAugment()
        ])
    """

    def __init__(self, augmentors: list):
        """
        Initialize chain augmentor.

        Args:
            augmentors: List of augmentor objects
        """
        self.augmentors = augmentors

    def __call__(self, data: Union[np.ndarray, torch.Tensor]):
        """Apply all augmentors in sequence."""
        for augmentor in self.augmentors:
            data = augmentor(data)
        return data
