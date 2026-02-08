"""
FastAPI Backend for DetectVoice

This API provides endpoints to upload audio files and get real/fake predictions,
and serves the frontend application.
"""
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
from pathlib import Path
import time
import yaml
import torch
import torchaudio
import librosa
import numpy as np

# Adjust the path to import from the 'src' and 'detectvoice_adversarial' directories
import sys
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.model_factory import create_model
from src.models.ensemble.master_ensemble import EnsembleModel
from detectvoice_adversarial.src.utils.audio import load_audio, AudioFeatureExtractor

# Global variables for configuration and model
CONFIG = None
MODEL = None
DEVICE = "cpu"
AUDIO_FEATURE_EXTRACTOR = None

# Create a temporary directory for uploads
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="DetectVoice API",
    description="API for real-time audio deepfake detection.",
    version="1.0.0"
)

# Configure CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_config(config_path='configs/config.yaml') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess_audio(audio_path: Path) -> torch.Tensor:
    global CONFIG, AUDIO_FEATURE_EXTRACTOR, DEVICE
    data_cfg = CONFIG.get('data', {})
    feature_cfg = data_cfg.get('features', {})

    if AUDIO_FEATURE_EXTRACTOR is None:
        AUDIO_FEATURE_EXTRACTOR = AudioFeatureExtractor(**data_cfg)

    waveform, sr = load_audio(audio_path, sample_rate=data_cfg.get('sample_rate', 16000), duration=data_cfg.get('duration', 3.0))
    feature_type = CONFIG.get('inference', {}).get('feature_type', 'mel_spectrogram')

    if feature_type == "mel_spectrogram":
        features = AUDIO_FEATURE_EXTRACTOR.extract_mel_spectrogram(waveform)
    else: # Add other feature types as needed
        features = AUDIO_FEATURE_EXTRACTOR.extract_mfcc(waveform)

    if data_cfg.get('normalization', {}).get('enabled', True):
        features = AUDIO_FEATURE_EXTRACTOR.normalize(features)

    if features.dim() == 2:
        features = features.unsqueeze(0).unsqueeze(0)
    
    return features.to(DEVICE)

def load_inference_model():
    global CONFIG, MODEL, DEVICE
    print("Loading inference configuration...")
    inference_cfg = CONFIG.get('inference', {})
    models_dir = CONFIG.get('paths', {}).get('models_dir', 'outputs/models')
    DEVICE = inference_cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {DEVICE}")

    # Simplified logic: always load a single model for now
    model_name = inference_cfg.get('single_model_name', 'cnn_2d')
    print(f"Loading model: {model_name}...")
    
    model_instance = create_model(model_name, inference_cfg.get('feature_type', 'mel_spectrogram'), CONFIG)
    model_path = Path(models_dir) / "best_model.pt"

    if model_path.exists():
        print(f"Loading weights from {model_path}...")
        state_dict = torch.load(model_path, map_location=DEVICE)['model_state_dict']
        model_instance.load_state_dict(state_dict)
        MODEL = model_instance.to(DEVICE).eval()
        print(f"Model '{model_name}' loaded successfully.")
    else:
        print(f"Warning: Model weights not found at {model_path}. API will run with a dummy model.")
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([[10.0, 0.0]], device=x.device)
        MODEL = DummyModel().to(DEVICE).eval()
        print("Fallback to generic dummy model for inference.")

    print("Model loading complete.")

@app.on_event("startup")
async def startup_event():
    global CONFIG
    CONFIG = load_config()
    load_inference_model()
    print("API startup complete.")

@app.post("/predict", tags=["Prediction"])
async def predict_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        features = preprocess_audio(file_path)

        with torch.no_grad():
            outputs = MODEL(features)
            probs = torch.softmax(outputs, dim=1)[0]
            is_fake_proba = probs[0].item() if len(probs) > 0 else 0.0
            is_real_proba = probs[1].item() if len(probs) > 1 else 0.0

            is_fake_prediction = is_fake_proba > is_real_proba
            confidence = max(is_fake_proba, is_real_proba)
        
        explanation = "Model explanation placeholder."

        return JSONResponse(content={
            "filename": file.filename,
            "is_fake": is_fake_prediction,
            "confidence_score": f"{confidence:.2f}",
            "explanation": explanation
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()

# --- Mount Static Files for Frontend ---
# This MUST be the last thing before the __main__ block
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    print("Starting DetectVoice API server...")
    print("Access the web UI at http://127.0.0.1:8000")
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=True)
