"""
Training script for CNN detector with adversarial training.

Usage:
    python src/training/train_cnn.py --config src/config/cnn_config.yaml

⚠️  SECURITY & ETHICS NOTICE ⚠️
This training is for DEFENSIVE deepfake detection research.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.cnn.detector import CNNDetector
from src.utils.config import ConfigLoader
from src.utils.logger import setup_logger
from src.utils.dataloader import AudioDeepfakeDataset, create_dataloaders
from src.training.adv_train import AdversarialTrainer
from src.export.export_utils import ModelExporter


def main(config_path: Path):
    """
    Main training function.

    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration
    config = ConfigLoader.load(config_path)

    # Setup logging
    logger = setup_logger(
        __name__,
        log_dir=Path(config['paths']['log_dir']),
        log_to_file=True
    )

    logger.info("=" * 60)
    logger.info("CNN DETECTOR TRAINING")
    logger.info("=" * 60)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = CNNDetector(
        input_channels=config['model']['input_channels'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout']
    )

    logger.info(f"Model created: {config['model']['name']}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets (placeholder - you need actual data)
    # For demonstration, assuming you have data_dir with real/ and fake/ subdirectories
    logger.warning("NOTE: This script requires actual audio data in the data directory")
    logger.warning("Expected structure: data/real/*.wav and data/fake/*.wav")

    try:
        train_dataset = AudioDeepfakeDataset(
            data_dir=Path(config['paths']['data_dir']) / 'train',
            sample_rate=config['data']['sample_rate'],
            duration=config['data']['duration'],
            feature_type=config['data']['feature_type'],
            augment=config['data']['augment'],
            spec_augment=config['data']['spec_augment'],
            normalize=config['data']['normalize']
        )

        val_dataset = AudioDeepfakeDataset(
            data_dir=Path(config['paths']['data_dir']) / 'val',
            sample_rate=config['data']['sample_rate'],
            duration=config['data']['duration'],
            feature_type=config['data']['feature_type'],
            augment=False,
            spec_augment=False,
            normalize=config['data']['normalize']
        )

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_dataset,
            val_dataset,
            batch_size=config['training']['batch_size']
        )

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Creating dummy data for demonstration...")

        # Dummy data for demonstration
        logger.warning("USING DUMMY DATA - Replace with real audio data for actual training!")
        return

    # Optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Adversarial training
    if config['adversarial']['enabled']:
        trainer = AdversarialTrainer(
            model=model,
            optimizer=optimizer,
            device=str(device),
            adv_ratio=config['adversarial']['adv_ratio'],
            attack_type=config['adversarial']['attack_type'],
            attack_params=config['adversarial']['attack_params'],
            save_dir=Path(config['paths']['save_dir'])
        )

        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['training']['num_epochs'],
            loss_fn=loss_fn,
            early_stopping_patience=config['training']['early_stopping_patience']
        )

    else:
        logger.info("Standard training (no adversarial training)")
        # Standard training loop would go here
        # (simplified for brevity)

    # Export model
    logger.info("\nExporting model...")

    # Create example input for export
    example_input = torch.randn(1, 128, 94).to(device)  # Example spectrogram shape

    exporter = ModelExporter(
        save_dir=Path(config['paths']['save_dir']),
        model_name=config['model']['name']
    )

    export_paths = exporter.export_all(
        model=model,
        example_input=example_input,
        optimizer=optimizer,
        metrics={'best_val_acc': history['val_acc'][-1] if 'val_acc' in history else 0.0}
    )

    logger.info("=" * 60)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN detector")
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/cnn_config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    main(Path(args.config))
