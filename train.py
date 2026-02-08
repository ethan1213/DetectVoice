"""
Main Training Runner Script
This script orchestrates the entire training process for a selected model.

Example usage:
    python train.py --model cnn_2d
    python train.py --model bilstm
"""
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np

# Adjust the path to import from the 'src' and 'detectvoice_adversarial' directories
import sys
from pathlib import Path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))
# No longer need to add subdirectories explicitly if imports are correct

from src.models.model_factory import create_model
from src.training.train_master import MasterTrainer
from detectvoice_adversarial.src.utils.dataloader import AudioDeepfakeDataset, create_dataloaders
from src.utils.download_datasets import DatasetManager

def load_config(config_path='configs/config.yaml') -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main(args):
    """Main training function."""
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    config = load_config()
    if not config:
        print("Error: Could not load configuration. Exiting.")
        return

    # --- 2. Prepare Dataset ---
    print("Preparing dataset...")
    # First, ensure data is available using the DatasetManager
    manager = DatasetManager(config)
    manager.prepare_dataset() # This will check local path or download from GDrive

    # Create the full dataset instance
    # The parameters are taken directly from the config file
    data_cfg = config.get('data', {})
    feature_cfg = data_cfg.get('features', {})
    aug_cfg = data_cfg.get('augmentation', {})
    
    # The dataloader expects the data to be in 'real' and 'fake' subfolders
    # Our data_source path points to the root of this structure
    dataset_path = Path(config['data_source']['local']['path'])

    full_dataset = AudioDeepfakeDataset(
        data_dir=dataset_path,
        sample_rate=data_cfg.get('sample_rate', 16000),
        duration=data_cfg.get('duration', 3.0),
        feature_type=args.feature_type, # Use feature type from args
        augment=aug_cfg.get('enabled', False),
        spec_augment=aug_cfg.get('spec_augment', {}).get('enabled', False), # Assumes a spec_augment config block
        normalize=data_cfg.get('normalization', {}).get('enabled', True) # Assumes a normalization config block
    )

    # Split dataset into training and validation sets
    total_size = len(full_dataset)
    train_split = data_cfg.get('train_split', 0.7)
    val_split = data_cfg.get('val_split', 0.15)
    
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size # The rest is for test

    print(f"Splitting dataset: {train_size} train, {val_size} validation, {test_size} test samples.")
    
    # Ensure splits don't overlap due to rounding
    if train_size + val_size + test_size > total_size:
        val_size -= (train_size + val_size + test_size - total_size)

    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.get('training', {}).get('batch_size', 32),
        num_workers=config.get('hardware', {}).get('num_workers', 4)
    )

    # --- 3. Create Model ---
    print(f"Creating model: {args.model}...")
    model = create_model(model_name=args.model, feature_type=args.feature_type, config=config)
    
    # --- 4. Initialize Trainer and Start Training ---
    print("Initializing trainer...")
    device = config.get('hardware', {}).get('device', 'cuda') if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = MasterTrainer(
        model=model,
        config=config,
        device=device
    )

    print("--- Starting Training ---")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )
    print("--- Training Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Master Training Runner for DetectVoice")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help="The name of the model to train (e.g., 'cnn_2d', 'bilstm'). Must match a key in the model registry and config."
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        default='mel_spectrogram',
        help="Feature type to use for training (e.g., 'mel_spectrogram', 'mfcc', 'waveform')."
    )
    
    # Add more arguments as needed, e.g., to override config values
    # parser.add_argument('--batch-size', type=int, help="Override batch size from config.")

    args = parser.parse_args()
    main(args)
