"""
Unified Inference Engine
Supports all model types with batch and single-file inference
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, Dict, List
import warnings

class InferenceEngine:
    """Universal inference engine for all model types"""
    def __init__(self, model, config: Dict, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        self.sample_rate = config.get('sample_rate', 16000)

    def load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(str(audio_path))

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform.squeeze(0)

    @torch.no_grad()
    def predict_single(self, audio_path: Union[str, Path]) -> Dict:
        """
        Predict single audio file

        Returns:
            Dict with 'prediction', 'confidence', 'probabilities'
        """
        waveform = self.load_audio(audio_path)
        waveform = waveform.unsqueeze(0).to(self.device)  # Add batch dim

        logits = self.model(waveform)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        conf = probs[0, pred].item()

        return {
            'prediction': 'FAKE' if pred == 0 else 'REAL',
            'prediction_label': pred,
            'confidence': conf,
            'probabilities': {
                'fake': probs[0, 0].item(),
                'real': probs[0, 1].item()
            }
        }

    @torch.no_grad()
    def predict_batch(self, audio_paths: List[Union[str, Path]]) -> List[Dict]:
        """Predict multiple audio files"""
        results = []
        for path in audio_paths:
            try:
                result = self.predict_single(path)
                result['file'] = str(path)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Error processing {path}: {e}")
                results.append({'file': str(path), 'error': str(e)})

        return results

    @torch.no_grad()
    def predict_realtime(self, waveform: torch.Tensor) -> Dict:
        """
        Realtime prediction for streaming audio

        Args:
            waveform: Audio tensor (already preprocessed)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)
        logits = self.model(waveform)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        return {
            'prediction': 'FAKE' if pred == 0 else 'REAL',
            'confidence': probs[0, pred].item(),
            'probabilities': probs[0].cpu().numpy()
        }

    def save_results(self, results: List[Dict], output_path: Union[str, Path]):
        """Save inference results to JSON"""
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

class EnsembleInferenceEngine:
    """Inference engine for ensemble models"""
    def __init__(self, models: Dict, config: Dict, device='cuda'):
        self.engines = {name: InferenceEngine(model, config, device) for name, model in models.items()}
        self.config = config

    @torch.no_grad()
    def predict_single(self, audio_path: Union[str, Path], strategy: str = 'avg') -> Dict:
        """Predict using ensemble"""
        all_results = {}
        all_probs = []

        for name, engine in self.engines.items():
            result = engine.predict_single(audio_path)
            all_results[name] = result
            all_probs.append([result['probabilities']['fake'], result['probabilities']['real']])

        # Aggregate predictions
        all_probs = np.array(all_probs)
        if strategy == 'avg':
            final_probs = all_probs.mean(axis=0)
        elif strategy == 'max':
            final_probs = all_probs.max(axis=0)
        else:
            final_probs = all_probs.mean(axis=0)

        final_pred = 'FAKE' if final_probs[0] > final_probs[1] else 'REAL'

        return {
            'prediction': final_pred,
            'confidence': max(final_probs),
            'probabilities': {'fake': final_probs[0], 'real': final_probs[1]},
            'individual_predictions': all_results
        }
