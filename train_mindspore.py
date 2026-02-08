"""
MindSpore Training Runner Script
Orchestrates the training process for a MindSpore model.

This script is a placeholder and requires a functional MindSpore installation
to be fully tested and used.

Example usage (on a machine with MindSpore installed):
    python train_mindspore.py
"""
import argparse
import yaml
from pathlib import Path
import numpy as np

# --- MindSpore Imports ---
# These will only work in an environment with mindspore installed.
try:
    import mindspore as ms
    from mindspore.dataset import GeneratorDataset
    from mindspore.train import Model, LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
    from mindspore.nn import CrossEntropyLoss, AdamWeightDecay, CosineDecayLR
    print("MindSpore imported successfully.")
except ImportError:
    print("Warning: MindSpore not found. This script is for structure and logic reference.")
    # Create dummy classes to allow the script to be parsed
    class Cell: pass
    class nn:
        Cell = Cell
    ms = type('mindspore', (), {'nn': nn})()
# --- End MindSpore Imports ---


# --- Project-specific Imports ---
# Assumes this script is run from the project root.
from src.models.mindspore.cnn_detector import CNNMindSpore
# Data handling will need a MindSpore-compatible dataset generator.
# We will use the existing PyTorch dataset as a basis for the generator.
from detectvoice_adversarial.src.utils.dataloader import AudioDeepfakeDataset


def load_config(config_path='configs/config.yaml') -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_ms_dataset(pytorch_dataset, batch_size, num_workers):
    """
    Creates a MindSpore GeneratorDataset from a PyTorch-style dataset.
    
    This acts as a bridge between our existing data loading logic and
    MindSpore's data pipeline.
    """
    # The 'source' argument of GeneratorDataset needs an iterable object.
    # The PyTorch dataset itself is such an object.
    dataset = GeneratorDataset(
        source=pytorch_dataset,
        column_names=["data", "label"],
        shuffle=True,
        num_parallel_workers=num_workers
    )
    
    # Apply batching
    dataset = dataset.batch(batch_size)
    
    return dataset

def main():
    """Main MindSpore training function."""
    print("--- MindSpore Training Runner ---")
    
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    config = load_config()
    ms_config = config.get('mindspore', {})
    if not ms_config.get('enabled', False):
        print("MindSpore training is disabled in config.yaml. Exiting.")
        return

    # --- 2. Prepare Dataset ---
    print("Preparing dataset (using PyTorch dataset as a source)...")
    data_cfg = config.get('data', {})
    dataset_path = Path(config['data_source']['local']['path'])

    # We use our existing PyTorch dataset class to handle file loading and feature extraction
    # This is simpler than re-implementing all the logic for MindSpore's dataset API.
    full_pytorch_dataset = AudioDeepfakeDataset(
        data_dir=dataset_path,
        sample_rate=data_cfg.get('sample_rate', 16000),
        duration=data_cfg.get('duration', 3.0),
        feature_type='mel_spectrogram' # Or make this an arg
    )
    
    if len(full_pytorch_dataset) == 0:
        print("\nERROR: Data directory is empty. Cannot proceed with training.")
        print(f"Please populate the '{dataset_path}' directory with 'real' and 'fake' subfolders.")
        return

    # Create MindSpore GeneratorDataset from the PyTorch dataset
    # Note: Train/val split needs to be handled. For simplicity, we use the whole dataset here.
    # In a real scenario, you would create two GeneratorDataset instances.
    ms_dataset = create_ms_dataset(
        pytorch_dataset=full_pytorch_dataset,
        batch_size=config.get('training', {}).get('batch_size', 32),
        num_workers=config.get('hardware', {}).get('num_workers', 1) # MS often uses 1 worker for GeneratorDataset
    )
    
    dataset_size = ms_dataset.get_dataset_size()
    print(f"MindSpore dataset created with {dataset_size} batches.")

    # --- 3. Create Model ---
    print(f"Creating MindSpore model: {ms_config['model_name']}...")
    model_params = ms_config.get(ms_config['model_name'], {})
    net = CNNMindSpore(**model_params)

    # --- 4. Define Optimizer, Loss, and Training Strategy ---
    print("Defining training strategy...")
    train_cfg = config.get('training', {})
    
    # Learning rate schedule
    lr_schedule = CosineDecayLR(
        min_lr=train_cfg['scheduler'].get('eta_min', 1e-6),
        max_lr=train_cfg.get('learning_rate', 1e-4),
        decay_epoch=train_cfg.get('epochs', 100)
    )

    # Optimizer
    optimizer = AdamWeightDecay(
        params=net.trainable_params(),
        learning_rate=lr_schedule,
        weight_decay=train_cfg.get('weight_decay', 0.01)
    )

    # Loss function
    loss_fn = CrossEntropyLoss(sparse=True, reduction='mean')

    # --- 5. Train Model ---
    # The 'Model' class is a high-level API for training
    model = Model(network=net, loss_fn=loss_fn, optimizer=optimizer, metrics={"accuracy"})

    print("--- Starting MindSpore Training ---")
    
    # Callbacks for monitoring
    loss_monitor = LossMonitor()
    time_monitor = TimeMonitor()
    
    # Checkpoint saving configuration
    checkpoint_cfg = CheckpointConfig(
        save_checkpoint_steps=ms_dataset.get_dataset_size(), # Save one checkpoint per epoch
        keep_checkpoint_max=5
    )
    checkpoint_cb = ModelCheckpoint(
        prefix=ms_config['model_name'],
        directory=str(Path(config['paths']['models_dir']) / 'mindspore'),
        config=checkpoint_cfg
    )
    
    # The train method runs the entire training loop
    model.train(
        epoch=train_cfg.get('epochs', 100),
        train_dataset=ms_dataset,
        callbacks=[loss_monitor, time_monitor, checkpoint_cb],
        dataset_sink_mode=True # Use data sinking for better performance on Ascend/GPU
    )

    print("--- MindSpore Training Finished ---")


if __name__ == '__main__':
    # A check to ensure MindSpore is available before running
    try:
        import mindspore
    except ImportError:
        print("FATAL: MindSpore is not installed. Please install it before running this script.")
    else:
        main()
