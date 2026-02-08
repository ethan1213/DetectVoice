"""
Dataset loaders and utilities for deepfake detection.

⚠️  SECURITY & ETHICS NOTICE ⚠️
This code is part of a research project for audio deepfake detection and adversarial robustness.
Data should be used in compliance with privacy laws and consent requirements.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import json
import random
from .audio import load_audio, AudioFeatureExtractor, AudioAugmentation, SpecAugment
from .logger import get_logger

logger = get_logger(__name__)


class AudioDeepfakeDataset(Dataset):
    """
    Dataset for audio deepfake detection.

    Expected structure:
        data/
            real/
                *.wav
            fake/
                *.wav

    Or use metadata JSON: [{"path": "...", "label": 0/1}, ...]
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
        sample_rate: int = 16000,
        duration: Optional[float] = 3.0,
        feature_type: str = "mel_spectrogram",
        augment: bool = False,
        spec_augment: bool = False,
        normalize: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory with real/ and fake/ subdirectories
            metadata_file: JSON file with list of {path, label} dicts
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
            feature_type: mel_spectrogram, spectrogram, mfcc, waveform
            augment: Apply audio augmentation
            spec_augment: Apply SpecAugment
            normalize: Normalize features
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.feature_type = feature_type
        self.normalize = normalize

        # Feature extractor
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

        # Augmentation
        self.augmentation = AudioAugmentation(sample_rate=sample_rate) if augment else None
        self.spec_augment = SpecAugment() if spec_augment else None

        # Load dataset
        self.samples = []
        if metadata_file is not None:
            self._load_from_metadata(metadata_file)
        elif data_dir is not None:
            self._load_from_directory(data_dir)
        else:
            raise ValueError("Either data_dir or metadata_file must be provided")

        logger.info(f"Loaded {len(self.samples)} audio samples")
        logger.info(f"Feature type: {feature_type}")

    def _load_from_directory(self, data_dir: Path) -> None:
        """Load samples from directory structure."""
        data_dir = Path(data_dir)

        # Real audio (label = 1)
        real_dir = data_dir / "real"
        if real_dir.exists():
            for ext in ['*.wav', '*.mp3', '*.flac']:
                for audio_file in real_dir.glob(ext):
                    self.samples.append({"path": str(audio_file), "label": 1})

        # Fake audio (label = 0)
        fake_dir = data_dir / "fake"
        if fake_dir.exists():
            for ext in ['*.wav', '*.mp3', '*.flac']:
                for audio_file in fake_dir.glob(ext):
                    self.samples.append({"path": str(audio_file), "label": 0})

    def _load_from_metadata(self, metadata_file: Path) -> None:
        """Load samples from metadata JSON file."""
        with open(metadata_file, 'r') as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        audio_path = Path(sample["path"])
        label = sample["label"]

        # Load audio
        waveform, sr = load_audio(
            audio_path,
            sample_rate=self.sample_rate,
            duration=self.duration
        )

        # Apply audio augmentation
        if self.augmentation is not None:
            waveform = self.augmentation.apply(waveform)

        # Extract features
        if self.feature_type == "waveform":
            features = waveform.squeeze(0)
        elif self.feature_type == "mel_spectrogram":
            features = self.feature_extractor.extract_mel_spectrogram(waveform)
            features = features.squeeze(0)
        elif self.feature_type == "spectrogram":
            features = self.feature_extractor.extract_spectrogram(waveform)
            features = features.squeeze(0)
        elif self.feature_type == "mfcc":
            features = self.feature_extractor.extract_mfcc(waveform)
            features = features.squeeze(0)
        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        # Apply SpecAugment
        if self.spec_augment is not None and len(features.shape) == 2:
            features = self.spec_augment.apply(features)

        # Normalize
        if self.normalize:
            features = self.feature_extractor.normalize(features)

        return features, label


class SiameseAudioDataset(Dataset):
    """Dataset for Siamese Network training with audio pairs."""

    def __init__(
        self,
        base_dataset: AudioDeepfakeDataset,
        num_pairs: int = 10000
    ):
        """
        Initialize Siamese dataset.

        Args:
            base_dataset: Base audio dataset
            num_pairs: Number of pairs to generate
        """
        self.base_dataset = base_dataset
        self.num_pairs = num_pairs
        self.pairs = self._generate_pairs()
        logger.info(f"Generated {len(self.pairs)} audio pairs")

    def _generate_pairs(self) -> List[Dict]:
        """Generate positive and negative pairs."""
        pairs = []

        # Get indices by label
        real_indices = [i for i, s in enumerate(self.base_dataset.samples) if s["label"] == 1]
        fake_indices = [i for i, s in enumerate(self.base_dataset.samples) if s["label"] == 0]

        for _ in range(self.num_pairs // 2):
            # Positive pair (same class)
            if random.random() < 0.5 and len(real_indices) >= 2:
                idx1, idx2 = random.sample(real_indices, 2)
                pairs.append({"idx1": idx1, "idx2": idx2, "label": 1})
            elif len(fake_indices) >= 2:
                idx1, idx2 = random.sample(fake_indices, 2)
                pairs.append({"idx1": idx1, "idx2": idx2, "label": 1})

            # Negative pair (different class)
            if len(real_indices) > 0 and len(fake_indices) > 0:
                idx1 = random.choice(real_indices)
                idx2 = random.choice(fake_indices)
                pairs.append({"idx1": idx1, "idx2": idx2, "label": 0})

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get a pair from the dataset."""
        pair = self.pairs[idx]
        features1, _ = self.base_dataset[pair["idx1"]]
        features2, _ = self.base_dataset[pair["idx2"]]
        label = pair["label"]
        return features1, features2, label


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")

    return train_loader, val_loader
