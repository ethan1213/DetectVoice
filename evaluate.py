"""
Master Evaluation Script

This script loads a trained model (single or ensemble) and evaluates its performance
on a test dataset, reporting detailed metrics and optionally generating plots.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust the path to import from the 'src' and 'detectvoice_adversarial' directories
import sys
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.model_factory import create_model
from src.models.ensemble.master_ensemble import EnsembleModel
from detectvoice_adversarial.src.utils.dataloader import AudioDeepfakeDataset, create_dataloaders
from detectvoice_adversarial.src.utils.audio import load_audio, AudioFeatureExtractor # For feature extraction


def load_config(config_path='configs/config.yaml') -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_trained_model(config: dict, device: str):
    """
    Loads a trained model (single or ensemble) based on config.
    """
    models_dir = Path(config.get('paths', {}).get('models_dir', 'outputs/models'))
    inference_cfg = config.get('inference', {})
    
    model = None

    # --- Load Ensemble Model ---
    if inference_cfg.get('ensemble_enabled', False):
        print("Loading ensemble model for evaluation...")
        ensemble_cfg = config.get('ensemble', {})
        
        # This part requires a list of trained base models to be loaded.
        # For evaluation, we need to know WHICH base models were part of the ensemble.
        # This implies a more structured way of saving/loading ensemble configuration.
        # For now, let's assume `ensemble.strategy` specifies which models.
        
        # Placeholder for real base model loading
        # In a real scenario, you would have a list of actual models from config
        base_models_config = {
            "cnn_2d": config.get('models', {}).get('deep_learning', {}).get('cnn_2d', {}),
            "bilstm": config.get('models', {}).get('deep_learning', {}).get('bilstm', {}),
        } # Example: specify relevant base models in config.yaml
        
        loaded_base_models = {}
        for model_name_key, model_cfg in base_models_config.items():
            print(f"Loading base model '{model_name_key}'...")
            base_model_instance = create_model(
                model_name_key, 
                inference_cfg.get('feature_type', 'mel_spectrogram'), 
                config
            )
            base_model_path = models_dir / f"{model_name_key}.pt"
            if base_model_path.exists():
                state_dict = torch.load(base_model_path, map_location=device)['model_state_dict']
                base_model_instance.load_state_dict(state_dict)
                base_model_instance.to(device).eval()
                loaded_base_models[model_name_key] = base_model_instance
            else:
                print(f"Warning: Weights for base model '{model_name_key}' not found at {base_model_path}. "
                      "This model will be skipped in the ensemble.")

        if not loaded_base_models:
            raise RuntimeError("No base models loaded for ensemble evaluation.")

        model = EnsembleModel(base_models=loaded_base_models, config=ensemble_cfg).to(device)

        if ensemble_cfg.get('strategy') == 'stacking':
            model.load_meta_model(models_dir=str(models_dir)) # Pass string path
            
        print("Ensemble model loaded.")


    # --- Load Single Model ---
    elif inference_cfg.get('single_model_enabled', True):
        single_model_name = inference_cfg.get('single_model_name', 'cnn_2d')
        print(f"Loading single model: {single_model_name} for evaluation...")
        
        model_instance = create_model(single_model_name, inference_cfg.get('feature_type', 'mel_spectrogram'), config)
        
        model_path = models_dir / f"best_model.pt" # MasterTrainer saves best model as 'best_model.pt'
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=device)['model_state_dict']
            model_instance.load_state_dict(state_dict)
            model_instance.to(device).eval()
            model = model_instance
            print(f"Model '{single_model_name}' weights loaded from {model_path}.")
        else:
            raise FileNotFoundError(f"Trained model weights not found at {model_path}. "
                                    "Please train a model first using train.py.")
    else:
        raise ValueError("No model (single or ensemble) enabled for evaluation in config.yaml")

    print("Model loading complete.")
    return model

def plot_confusion_matrix(cm, classes, output_dir, filename="confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    
    output_path = Path(output_dir) / filename
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")
    plt.close()


def main(args):
    """Main evaluation function."""
    print("--- Starting Model Evaluation ---")
    
    # --- 1. Load Configuration ---
    print("Loading configuration...")
    config = load_config()
    eval_cfg = config.get('evaluation', {})
    
    # --- 2. Setup Device ---
    device = config.get('hardware', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation device: {device}")

    # --- 3. Load Trained Model ---
    model = load_trained_model(config, device)
    
    # --- 4. Prepare Test Dataset ---
    print("Preparing test dataset...")
    data_cfg = config.get('data', {})
    dataset_path = Path(config['data_source']['local']['path'])

    # Test dataset will be the 'test_split' from the overall dataset
    full_dataset = AudioDeepfakeDataset(
        data_dir=dataset_path,
        sample_rate=data_cfg.get('sample_rate', 16000),
        duration=data_cfg.get('duration', 3.0),
        feature_type=config.get('inference', {}).get('feature_type', 'mel_spectrogram'),
        augment=False, # No augmentation for evaluation
        normalize=data_cfg.get('normalization', {}).get('enabled', True)
    )

    total_size = len(full_dataset)
    train_split = data_cfg.get('train_split', 0.7)
    val_split = data_cfg.get('val_split', 0.15)
    test_split = data_cfg.get('test_split', 0.15)

    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size # The rest is for test

    print(f"Splitting dataset for evaluation: {test_size} test samples.")
    if test_size <= 0:
        raise ValueError("Test dataset is empty. Ensure your data directory has samples and split ratios are correct.")
    
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    test_loader, _ = create_dataloaders(
        train_dataset=torch.utils.data.Subset(full_dataset, []), # Dummy train_dataset
        val_dataset=test_dataset, # Use test_dataset as val_dataset for create_dataloaders
        batch_size=config.get('inference', {}).get('batch_size', 1),
        num_workers=config.get('hardware', {}).get('num_workers', 4),
        pin_memory=config.get('hardware', {}).get('pin_memory', True)
    )
    
    # --- 5. Run Evaluation ---
    print("Running evaluation on test set...")
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy()[:, 1]) # Prob of positive class (real)

    # --- 6. Calculate and Report Metrics ---
    print("\n--- Evaluation Results ---")
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    # --- 7. Save Plots (optional) ---
    plot_dir = Path(config.get('paths', {}).get('plots_dir', 'outputs/plots'))
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, classes=['Fake', 'Real'], output_dir=plot_dir)
    
    print("--- Evaluation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DetectVoice Model Evaluation Script")
    # You might want to add arguments to override config values for evaluation
    args = parser.parse_args()
    main(args)
