"""
Spectral and temporal perturbations specific to audio deepfake detection.

These attacks exploit audio-specific properties:
- Spectral magnitude perturbations
- Time warping
- Phase perturbations
- Low-amplitude noise with SNR constraints

⚠️  SECURITY & ETHICS NOTICE ⚠️
This implementation is for DEFENSIVE research purposes only.
DO NOT use for malicious purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
from src.utils.audio import compute_snr, AudioFeatureExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SpectralPerturbation:
    """
    Perturbations in the spectrogram domain.

    Applies perturbations to mel-spectrogram magnitude while
    maintaining perceptual quality constraints.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.05,
        num_iter: int = 10,
        alpha: float = 0.01,
        targeted: bool = False
    ):
        """
        Initialize spectral perturbation attack.

        Args:
            model: Target model (operates on spectrograms)
            epsilon: Maximum perturbation in spectrogram domain
            num_iter: Number of iterations
            alpha: Step size
            targeted: Targeted attack flag
        """
        self.model = model
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.alpha = alpha
        self.targeted = targeted

        self.model.eval()

    def generate(
        self,
        x_spec: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarial spectrogram.

        Args:
            x_spec: Input spectrogram (batch, freq, time) or (batch, 1, freq, time)
            y: True labels
            loss_fn: Loss function

        Returns:
            Tuple of (adversarial spectrogram, metrics)
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        x_spec = x_spec.detach().clone()
        x_adv = x_spec.clone()

        # Random initialization
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-self.epsilon, self.epsilon)
        x_adv = torch.clamp(x_adv, x_spec - self.epsilon, x_spec + self.epsilon)

        # Iterative attack
        for i in range(self.num_iter):
            x_adv.requires_grad = True

            outputs = self.model(x_adv)

            if self.targeted:
                loss = -loss_fn(outputs, y)
            else:
                loss = loss_fn(outputs, y)

            self.model.zero_grad()
            loss.backward()

            grad = x_adv.grad.detach()

            # Update
            x_adv = x_adv.detach() + self.alpha * grad.sign()

            # Project to epsilon ball
            perturbation = torch.clamp(x_adv - x_spec, -self.epsilon, self.epsilon)
            x_adv = x_spec + perturbation

        # Metrics
        perturbation = (x_adv - x_spec).detach()
        l2_norm = torch.norm(perturbation.view(x_spec.size(0), -1), p=2, dim=1).mean().item()

        metrics = {
            'l2_norm': l2_norm,
            'epsilon': self.epsilon,
            'num_iter': self.num_iter
        }

        return x_adv.detach(), metrics


class TimeWarping:
    """
    Time warping attack for audio.

    Applies small time-domain warping/stretching to audio
    to evade detection while maintaining perceptual quality.
    """

    def __init__(
        self,
        warp_factor_range: Tuple[float, float] = (0.95, 1.05)
    ):
        """
        Initialize time warping.

        Args:
            warp_factor_range: Range of warping factors (speed change)
        """
        self.warp_factor_range = warp_factor_range

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to waveform.

        Args:
            waveform: Input waveform (batch, channels, time) or (channels, time)

        Returns:
            Warped waveform
        """
        was_batched = waveform.dim() == 3

        if not was_batched:
            waveform = waveform.unsqueeze(0)

        batch_size = waveform.size(0)
        warped_batch = []

        for i in range(batch_size):
            warp_factor = np.random.uniform(
                self.warp_factor_range[0],
                self.warp_factor_range[1]
            )

            # Resample using interpolation
            wave = waveform[i]
            original_length = wave.size(-1)
            new_length = int(original_length / warp_factor)

            # Use interpolation to warp
            wave_warped = F.interpolate(
                wave.unsqueeze(0),
                size=new_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)

            # Pad or crop to original length
            if new_length > original_length:
                wave_warped = wave_warped[..., :original_length]
            elif new_length < original_length:
                padding = original_length - new_length
                wave_warped = F.pad(wave_warped, (0, padding))

            warped_batch.append(wave_warped)

        result = torch.stack(warped_batch, dim=0)

        if not was_batched:
            result = result.squeeze(0)

        return result


class LowAmplitudeNoise:
    """
    Add low-amplitude noise with SNR constraints.

    Useful for testing robustness to subtle perturbations.
    """

    def __init__(
        self,
        target_snr_db: float = 30.0,
        noise_type: str = 'gaussian'
    ):
        """
        Initialize low-amplitude noise attack.

        Args:
            target_snr_db: Target SNR in dB
            noise_type: Type of noise ('gaussian', 'uniform')
        """
        self.target_snr_db = target_snr_db
        self.noise_type = noise_type

    def apply(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add low-amplitude noise to waveform.

        Args:
            waveform: Input waveform

        Returns:
            Noisy waveform
        """
        # Generate noise
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(waveform)
        elif self.noise_type == 'uniform':
            noise = torch.rand_like(waveform) * 2 - 1
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")

        # Compute signal power
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)

        # Scale noise to achieve target SNR
        target_noise_power = signal_power / (10 ** (self.target_snr_db / 10))
        noise_scaled = noise * torch.sqrt(target_noise_power / (noise_power + 1e-8))

        return waveform + noise_scaled


class FrequencyMasking:
    """
    Adversarial frequency masking attack.

    Selectively masks frequency bands to evade detection.
    """

    def __init__(
        self,
        model: nn.Module,
        num_masks: int = 2,
        mask_width: int = 10,
        num_iter: int = 5
    ):
        """
        Initialize frequency masking attack.

        Args:
            model: Target model
            num_masks: Number of frequency masks
            mask_width: Width of each mask
            num_iter: Iterations to optimize mask positions
        """
        self.model = model
        self.num_masks = num_masks
        self.mask_width = mask_width
        self.num_iter = num_iter

        self.model.eval()

    def generate(
        self,
        x_spec: torch.Tensor,
        y: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate adversarially masked spectrogram.

        Args:
            x_spec: Input spectrogram
            y: True labels
            loss_fn: Loss function

        Returns:
            Tuple of (masked spectrogram, metrics)
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        x_spec = x_spec.detach().clone()
        freq_bins = x_spec.size(-2)

        best_x_adv = x_spec.clone()
        best_loss = float('-inf')

        # Try different mask positions
        for _ in range(self.num_iter):
            x_adv = x_spec.clone()

            for _ in range(self.num_masks):
                # Random frequency band
                freq_start = np.random.randint(0, max(1, freq_bins - self.mask_width))
                freq_end = min(freq_start + self.mask_width, freq_bins)

                # Apply mask (set to zero or mean)
                mask_value = x_adv[..., freq_start:freq_end, :].mean()
                x_adv[..., freq_start:freq_end, :] = mask_value

            # Evaluate
            with torch.no_grad():
                outputs = self.model(x_adv)
                loss = -loss_fn(outputs, y)

                if loss.item() > best_loss:
                    best_loss = loss.item()
                    best_x_adv = x_adv.clone()

        metrics = {
            'num_masks': self.num_masks,
            'mask_width': self.mask_width
        }

        return best_x_adv, metrics


def apply_random_spectral_perturbation(
    spectrogram: torch.Tensor,
    epsilon: float = 0.05
) -> torch.Tensor:
    """
    Apply random spectral perturbation.

    Args:
        spectrogram: Input spectrogram
        epsilon: Perturbation magnitude

    Returns:
        Perturbed spectrogram
    """
    noise = torch.randn_like(spectrogram) * epsilon
    return spectrogram + noise
