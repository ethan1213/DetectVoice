# 🎤 DetectVoice v2.0: Advanced Audio Deepfake Detection Suite

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Estado del arte en detección de deepfakes de audio con arquitecturas múltiples y robustez adversarial**

[Características](#-características) •
[Instalación](#-instalación) •
[Uso Rápido](#-uso-rápido) •
[Modelos](#-arquitecturas) •
[Datasets](#-datasets) •
[Disclaimer](#-disclaimer-legal)

</div>

---

## ⚠️ DISCLAIMER LEGAL Y ÉTICO

### 🚨 USO RESPONSABLE

**✅ PERMITIDO:**
- Protección contra fraudes y estafas
- Investigación académica
- Verificación de autenticidad
- Desarrollo de seguridad
- Auditoría autorizada
- Educación en ML/cybersecurity

**❌ PROHIBIDO:**
- Vigilancia no autorizada
- Identificación sin consentimiento
- Violación de privacidad
- Discriminación
- Acoso o chantaje
- Uso ilegal

### 📜 Responsabilidades

El uso de este software implica aceptación de cumplir con GDPR, CCPA, y leyes locales de privacidad. Los usuarios son TOTALMENTE RESPONSABLES del uso legal y ético. Los autores NO son responsables de mal uso.

---

## 🎯 Características Principales

### 🏗️ 25+ Modelos Implementados

- **Transformers**: Wav2Vec2, HuBERT, AST
- **Clásicos**: SVM, XGBoost, Random Forest, Logistic Regression
- **Deep Learning**: CNN 1D/2D, LSTM, BiLSTM, GRU, BiGRU, CRNN
- **Avanzados**: ECAPA-TDNN, ResNet-Audio, QuartzNet, Conformer, Harmonic CNN
- **Generativos**: Autoencoders, VAE, Siamese Networks, GAN Discriminators

### 🎼 Extracción de Características

- MFCC, Mel Spectrogram, STFT, Chroma, CQT
- Spectral features, Zero Crossing Rate
- Pitch & Formant tracking
- Raw waveform support

### 🔄 Data Augmentation

- Noise injection (white/pink/brown)
- Time stretching, Pitch shifting
- Codec simulation, Room acoustics
- SpecAugment

### 🎯 Sistema de Ensamblado

- Simple/Weighted Averaging
- Voting (hard/soft)
- Stacking con meta-modelo
- Jury System (N-model agreement)

### 📊 Tracking & Visualización

- TensorBoard & MLflow integration
- ROC, PR, DET curves
- Confusion matrices
- t-SNE/UMAP embeddings
- Feature importance

---

## 📦 Instalación

```bash
# Clonar
git clone https://github.com/ethan1213/DetectVoice.git
cd DetectVoice

# Opción 1: pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Opción 2: conda
conda env create -f environment.yaml
conda activate detectvoice
```

---

## 🚀 Uso Rápido

### Inferencia

```python
from src.inference.inference_engine import InferenceEngine
from src.models.transformers import Wav2Vec2Detector

model = Wav2Vec2Detector()
engine = InferenceEngine(model, {'sample_rate': 16000}, device='cuda')

result = engine.predict_single('test.wav')
print(f"Predicción: {result['prediction']} (confianza: {result['confidence']:.2%})")
```

### Entrenamiento

```python
from src.training.train_master import MasterTrainer
from src.models.advanced import ECAPATDNNDetector

model = ECAPATDNNDetector()
trainer = MasterTrainer(model, config, device='cuda')
trainer.train(train_loader, val_loader, epochs=50)
```

---

## 🏛️ Arquitecturas Implementadas

| Categoría | Modelos | Papers |
|-----------|---------|--------|
| **Transformers** | Wav2Vec2, HuBERT, AST | Baevski+ 2020, Hsu+ 2021, Gong+ 2021 |
| **Avanzados** | ECAPA-TDNN, ResNet, QuartzNet, Conformer, Harmonic CNN | Desplanques+ 2020, He+ 2015, etc. |
| **Deep Learning** | CNN 1D/2D, LSTM, BiLSTM, GRU, BiGRU, CRNN | Standard architectures |
| **Clásicos** | SVM, XGBoost, RF, LogReg | Standard ML |

---

## 📚 Datasets Soportados

### Deepfake
- ASVspoof 2019 & 2021
- FakeAVCeleb, WaveFake
- FoR, ADD 2022
- AUDETER, DSD-Corpus

### Real Voice
- LibriSpeech, LibriTTS-R
- Mozilla Common Voice
- VoxCeleb 1 & 2

### Scripts de Descarga

```bash
python src/utils/download_datasets.py
```

---

## 🛠️ Estructura del Proyecto

```
DetectVoice/
├── configs/config.yaml
├── src/
│   ├── preprocessing/
│   ├── models/
│   │   ├── transformers/
│   │   ├── classical/
│   │   ├── deep_learning/
│   │   ├── advanced/
│   │   └── ensemble/
│   ├── training/
│   ├── visualization/
│   ├── inference/
│   └── utils/
├── data/
├── outputs/
└── weights/
```

---

## 📄 Licencia

MIT License con restricciones éticas. Ver LICENSE para detalles.

---

## 🙏 Referencias

```bibtex
@inproceedings{baevski2020wav2vec,
  title={wav2vec 2.0: A framework for self-supervised learning of speech representations},
  author={Baevski, Alexei and others},
  booktitle={NeurIPS},
  year={2020}
}
```

---

<div align="center">

**⚖️ USE CON RESPONSABILIDAD | Desarrollado para combatir deepfakes**

</div>
