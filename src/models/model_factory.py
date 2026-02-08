"""
Model Factory
Dynamically creates and returns a model instance based on configuration.
"""
from typing import Dict
import torch.nn as nn

# Import all model classes
from src.models.deep_learning.cnn_models import CNN1D, CNN2D
from src.models.deep_learning.rnn_models import LSTMDetector, BiLSTMDetector, GRUDetector, BiGRUDetector
# Add other model imports here as they are created/refactored

# A mapping from config names to model classes
MODEL_REGISTRY = {
    "cnn_1d": CNN1D,
    "cnn_2d": CNN2D,
    "lstm": LSTMDetector,
    "bilstm": BiLSTMDetector,
    "gru": GRUDetector,
    "bigru": BiGRUDetector,
}

def get_input_size(feature_type: str, config: Dict) -> int:
    """Calculates the input size for a model based on the feature type."""
    feature_config = config.get('data', {}).get('features', {})
    if feature_type == 'mel_spectrogram':
        return feature_config.get('n_mels', 128)
    elif feature_type == 'mfcc':
        return feature_config.get('n_mfcc', 40)
    elif feature_type == 'spectrogram':
        return feature_config.get('n_fft', 2048) // 2 + 1
    else:
        # For waveforms or unknown types, we don't need a specific input size
        # as the model (e.g., CNN1D) handles it via in_channels.
        return 1


def create_model(model_name: str, feature_type: str, config: Dict) -> nn.Module:
    """
    Creates a model instance from the registry based on the model name and config.

    Args:
        model_name (str): The name of the model to create.
        feature_type (str): The type of feature used, to determine model input size.
        config (Dict): The main configuration dictionary.

    Returns:
        nn.Module: An instance of the requested model.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'. Available models are: {list(MODEL_REGISTRY.keys())}")

    model_class = MODEL_REGISTRY[model_name]
    
    # Find the config block for the specific model
    model_config = {}
    for category in ['deep_learning', 'transformers', 'advanced', 'classical', 'generative']:
        if model_name in config.get('models', {}).get(category, {}).get('models', []):
            model_config = config.get('models', {}).get(category, {}).get(model_name, {})
            break
            
    if not model_config:
        print(f"Warning: No specific configuration found for '{model_name}'. Using model defaults.")
        
    # For RNNs and other models that need it, calculate and add input_size
    if model_name in ["lstm", "bilstm", "gru", "bigru"]:
        model_config['input_size'] = get_input_size(feature_type, config)

    # Create the model instance with its specific config
    try:
        model = model_class(**model_config)
        print(f"Successfully created model '{model_name}' with config: {model_config}")
        return model
    except TypeError as e:
        print(f"Error: Could not instantiate model '{model_name}'.")
        print(f"Check if the parameters in 'configs/config.yaml' match the model's __init__ method.")
        print(f"Original error: {e}")
        raise