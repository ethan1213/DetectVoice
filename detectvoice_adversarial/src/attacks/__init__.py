"""
Adversarial attacks module for DetectVoice.

   SECURITY & ETHICS NOTICE  
These attacks are for DEFENSIVE research purposes only.
"""

from .fgsm import FGSM, fgsm_attack
from .pgd import PGD, pgd_attack
from .cw import CarliniWagnerL2, cw_l2_attack
from .deepfool import DeepFool, deepfool_attack
from .spec_perturbations import (
    SpectralPerturbation,
    TimeWarping,
    LowAmplitudeNoise,
    FrequencyMasking,
    apply_random_spectral_perturbation
)

__all__ = [
    'FGSM', 'fgsm_attack',
    'PGD', 'pgd_attack',
    'CarliniWagnerL2', 'cw_l2_attack',
    'DeepFool', 'deepfool_attack',
    'SpectralPerturbation',
    'TimeWarping',
    'LowAmplitudeNoise',
    'FrequencyMasking',
    'apply_random_spectral_perturbation'
]
