# DetectVoice
Grupo de IAs que identifican voces falsas

---

## 🚀 DetectVoice - Sistema de Detección de Deepfakes de Voz

DetectVoice es un sistema profesional de detección de deepfakes de audio basado en múltiples modelos de deep learning. Incluye capacidades de entrenamiento, evaluación, exportación y robustez adversarial.

### 📋 Características Principales

- **Múltiples Modelos de Detección**: CNN, RNN, CRNN, Transformer
- **Exportación Multi-formato**: PyTorch (.pt), TorchScript (.ts), ONNX (.onnx)
- **Congelación de Modelos**: Función `freeze_model()` para deployment
- **Métricas Automáticas**: Generación de gráficos y reportes completos
- **Módulo Adversarial Opcional**: Testing de robustez (FGSM, PGD, C&W, DeepFool)
- **Evaluación Adversarial**: Reportes detallados de robustez

### 📂 Estructura del Proyecto

```
DetectVoice/
├── src/
│   ├── models/          # Modelos de detección (CNN, RNN, etc.)
│   ├── training/        # Scripts de entrenamiento
│   ├── evaluation/      # Módulos de evaluación
│   ├── utils/           # Utilidades (audio, métricas, export)
│   ├── data/            # Data loaders
│   └── adversarial/     # Módulo adversarial OPCIONAL
├── reports/
│   ├── plots/           # Gráficos generados por modelo
│   └── adversarial/     # Reportes adversariales
├── checkpoints/
│   ├── pt/              # Checkpoints PyTorch
│   ├── torchscript/     # Modelos TorchScript
│   └── onnx/            # Modelos ONNX
├── datasets/            # Datasets de entrenamiento
└── configs/             # Archivos de configuración
```

### 🎯 Uso Rápido

#### Entrenamiento de Modelo

```python
from src.models.cnn_detector import CNNDetector
from src.utils.metrics import MetricsLogger

# Crear modelo
model = CNNDetector(input_channels=1, num_classes=2)

# Entrenar (tu código de entrenamiento existente)
# ...

# Generar métricas automáticamente
metrics_logger = MetricsLogger(model_name="CNN", save_dir="reports/plots")
metrics_logger.generate_all_plots(y_true, y_pred, y_scores, real_spec, fake_spec)
```

#### Exportación de Modelo

```python
# Congelar modelo para deployment
model.freeze()

# Exportar a todos los formatos
example_input = torch.randn(1, 128, 94)  # Ejemplo de input
paths = model.export_all(
    base_path="checkpoints",
    model_name="CNN_detector",
    example_input=example_input
)
# Genera: .pt, .ts, .onnx automáticamente
```

#### Evaluación Adversarial (Opcional)

```python
from src.adversarial.adversarial_evaluator import AdversarialEvaluator

# Crear evaluador
evaluator = AdversarialEvaluator(
    model=model,
    model_name="CNN_detector",
    device='cuda'
)

# Evaluar robustez contra todos los ataques
results = evaluator.evaluate_all_attacks(test_loader)
# Genera reportes automáticos en reports/adversarial/
```

#### Uso de Ataques Adversariales

```python
from src.adversarial.fgsm import fgsm_attack
from src.adversarial.pgd import PGD

# FGSM
adv_audio = fgsm_attack(model, waveform, label, epsilon=0.03)

# PGD
pgd = PGD(model, epsilon=0.03, alpha=0.01, num_iter=10)
adv_audio, metrics = pgd.generate(waveform, label)
```

---

## 📊 Datasets Recomendados

Los siguientes datasets públicos son recomendados para entrenar y evaluar DetectVoice:

### Datasets Principales

1. **ASVspoof 2019**
   Dataset estándar para detección de audio spoofing
   🔗 https://www.asvspoof.org/index2019.html

2. **DSD-Corpus**
   Deepfake Speech Detection Corpus
   🔗 https://zenodo.org/records/13788455

3. **LibriTTS**
   Multi-speaker English corpus
   🔗 https://us.openslr.org/60/

4. **LibriTTS-R**
   Restored version of LibriTTS
   🔗 https://www.openslr.org/141/

### Datasets Especializados

5. **AUDETER**
   Audio Deepfake Detection Dataset
   🔗 https://arxiv.org/abs/2509.04345

6. **FoR Dataset**
   Fake or Real Dataset for Synthetic Speech Detection
   🔗 https://cisaad.umbc.edu/for-fake-or-real-dataset-for-synthetic-speech-detection/

7. **DEEP-VOICE**
   Deepfake Voice Recognition Dataset
   🔗 https://huggingface.co/datasets/DynamicSuperb/DeepFakeVoiceRecognition_DEEP-VOICE

8. **HAD (Half-Truth Audio Dataset)**
   Partially synthetic audio dataset
   🔗 https://arxiv.org/abs/2104.03617

9. **ELAD-SVDSR**
   Enhanced Dataset for Synthetic Voice Detection
   🔗 https://arxiv.org/abs/2510.00218

---

## 🎨 Métricas y Gráficos Generados Automáticamente

Cada entrenamiento genera automáticamente:

- ✅ **Curva de Pérdida** (loss_curve.png)
- ✅ **Curva de Accuracy** (accuracy_curve.png)
- ✅ **Curva ROC** (roc_curve.png)
- ✅ **Curva Precision-Recall** (precision_recall_curve.png)
- ✅ **Matriz de Confusión** (confusion_matrix.png)
- ✅ **Comparación Real vs Fake** (spectrogram_comparison.png)
- ✅ **Reporte de Clasificación** (classification_report.txt)
- ✅ **Métricas JSON** (metrics.json)

Todos los archivos se guardan en: `reports/plots/[modelo]/`

---

## 🔬 Mejoras del Sistema v2

### Nuevas Funcionalidades

1. **Sistema de Exportación Completo**
   - Exportación a PyTorch, TorchScript y ONNX
   - Función `freeze_model()` para deployment
   - Validación automática de modelos exportados

2. **Generación Automática de Métricas**
   - 6 tipos de gráficos generados automáticamente
   - Reportes en JSON y texto
   - Comparaciones visuales de espectrogramas

3. **Módulo Adversarial Opcional**
   - Ataques: FGSM, PGD, C&W, DeepFool
   - No interfiere con entrenamiento regular
   - Evaluación de robustez completa

4. **Evaluador Adversarial**
   - Prueba automática de todos los ataques
   - Reportes detallados con gráficos
   - Comparación de accuracy clean vs adversarial

5. **Compatibilidad Mejorada**
   - Integración con código existente
   - Sin cambios en flujo de entrenamiento
   - Funciones opcionales y modulares

### Mejoras de Código

- ✅ Tipado completo con type hints
- ✅ Logging profesional
- ✅ Documentación completa
- ✅ Estructura modular y extensible
- ✅ Compatible con GPU/CPU

---

## ⚠️ Disclaimer Legal y Ético

**Este proyecto tiene fines exclusivamente académicos, científicos y de ciberseguridad defensiva.**

Ningún componente del repositorio debe utilizarse para generar deepfakes o para actividades que involucren suplantación, ingeniería social o prácticas maliciosas.

**El uso indebido de este software es responsabilidad única del usuario.**

### Usos Permitidos
- ✅ Investigación académica en detección de deepfakes
- ✅ Desarrollo de sistemas de seguridad defensivos
- ✅ Pruebas de robustez de modelos
- ✅ Análisis forense de audio

### Usos Prohibidos
- ❌ Generación de deepfakes maliciosos
- ❌ Suplantación de identidad
- ❌ Fraude o engaño
- ❌ Violación de privacidad
- ❌ Cualquier uso ilegal o no ético

**Al usar este software, acepta cumplir con todas las leyes aplicables y utilizarlo únicamente para fines legítimos y éticos.**

---

## 📦 Instalación

```bash
# Clonar repositorio
git clone https://github.com/ethan1213/DetectVoice.git
cd DetectVoice

# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias Principales

- Python 3.8+
- PyTorch >= 1.10
- TorchAudio
- NumPy, SciPy
- Matplotlib, Seaborn
- scikit-learn
- ONNX (para exportación)

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas siguiendo estos principios:

1. Enfoque en detección y seguridad defensiva
2. Código bien documentado y testeado
3. Respeto a principios éticos
4. Compatibilidad con sistema existente

---

## 📝 Licencia

Este proyecto está disponible para fines de investigación y educación. Ver LICENSE para más detalles.

---

## 📧 Contacto

Para preguntas sobre uso responsable o colaboraciones de investigación, abrir un issue en GitHub.

---

**Desarrollado con el objetivo de mejorar la seguridad y autenticidad del audio digital.**

**Uso responsable y ético únicamente. 🔒**
# DetectVoice Adversarial Suite

**Professional Audio Deepfake Detection with Adversarial Robustness**

---

## ⚠️ CRITICAL SECURITY AND ETHICS NOTICE

### Purpose and Intended Use

This project is designed **EXCLUSIVELY for DEFENSIVE purposes**:

✅ **PERMITTED USES:**
- Academic research on deepfake detection
- Testing and improving detector robustness
- Security auditing and forensic analysis
- Training robust deepfake detection systems
- Defensive machine learning research

❌ **PROHIBITED USES:**
- Creating realistic audio deepfakes
- Unauthorized voice cloning or impersonation
- Malicious synthesis or deception
- Bypassing security systems
- Any illegal or unethical applications

### Legal and Ethical Responsibilities

**BY USING THIS CODE, YOU AGREE TO:**

1. **Use Responsibly**: Only use for legitimate defensive research and security purposes
2. **Comply with Laws**: Follow all applicable local, state, and federal laws
3. **Respect Privacy**: Obtain proper consent before processing voice data
4. **No Malicious Use**: Never use for creating undetectable deepfakes or impersonation
5. **Attribution**: Cite this work appropriately in academic research

**DISCLAIMER**: The authors and contributors are NOT responsible for misuse of this software. Users bear full legal and ethical responsibility for their actions.

### Toy Generator Limitations

The included "toy generator" is **INTENTIONALLY LIMITED**:
- Produces LOW-FIDELITY audio unsuitable for realistic synthesis
- Designed ONLY for testing discriminators
- NOT capable of high-quality voice cloning
- Includes obvious artifacts for detection

This is NOT a production TTS system and should not be used as such.

---

## 🎯 Project Overview

DetectVoice Adversarial Suite is a comprehensive framework for training, evaluating, and deploying robust audio deepfake detectors. It includes:

- **Multiple Detection Models**: CNN, RNN, CRNN, Transformer, Autoencoder, Siamese
- **Adversarial Robustness**: FGSM, PGD, C&W, DeepFool attacks
- **GAN Discriminators**: For forensic analysis
- **Ensemble Detection**: With explainability
- **Comprehensive Evaluation**: Metrics, plots, and reports
- **Model Export**: PyTorch, TorchScript, ONNX

---

## 📁 Project Structure

```
detectvoice_adversarial/
├── data/                          # Data directory
├── src/
│   ├── models/                    # Detection models
│   │   ├── cnn/                   # CNN detector
│   │   ├── rnn/                   # RNN (LSTM/GRU) detector
│   │   ├── crnn/                  # CRNN detector
│   │   ├── transformer/           # Transformer detector
│   │   ├── autoencoder/           # Autoencoder detector
│   │   ├── siamese/               # Siamese network
│   │   ├── discriminator/         # GAN discriminators (forensics)
│   │   ├── toy_generator/         # Low-fidelity toy generator (testing only)
│   │   └── ensemble/              # Ensemble detector
│   ├── attacks/                   # Adversarial attacks
│   │   ├── fgsm.py               # Fast Gradient Sign Method
│   │   ├── pgd.py                # Projected Gradient Descent
│   │   ├── cw.py                 # Carlini & Wagner
│   │   ├── deepfool.py           # DeepFool
│   │   └── spec_perturbations.py # Spectral/temporal perturbations
│   ├── training/                  # Training scripts
│   │   ├── train_cnn.py
│   │   └── adv_train.py          # Adversarial training
│   ├── evaluation/                # Evaluation suite
│   │   └── robustness_eval.py    # Robustness evaluation
│   ├── export/                    # Model export utilities
│   │   └── export_utils.py       # PT, TorchScript, ONNX export
│   ├── utils/                     # Utilities
│   │   ├── audio.py              # Audio processing
│   │   ├── dataloader.py         # Dataset loaders
│   │   ├── config.py             # Config management
│   │   └── logger.py             # Logging
│   └── config/                    # Configuration files
│       └── cnn_config.yaml
├── artifacts/                     # Output directory
│   ├── models/                    # Saved models
│   ├── metrics/                   # Metrics and reports
│   ├── plots/                     # Visualizations
│   └── adversarial_examples/      # Adversarial samples
├── notebooks/                     # Jupyter notebooks
├── tests/                         # Unit tests
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/detectvoice_adversarial.git
cd detectvoice_adversarial

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Prepare Data

Organize your audio data:

```
data/
├── train/
│   ├── real/
│   │   ├── audio1.wav
│   │   └── ...
│   └── fake/
│       ├── audio1.wav
│       └── ...
└── val/
    ├── real/
    └── fake/
```

### Train a Detector

```bash
# Train CNN detector with adversarial training
python src/training/train_cnn.py --config src/config/cnn_config.yaml
```

### Evaluate Robustness

```python
from src.evaluation.robustness_eval import RobustnessEvaluator
from src.models.cnn.detector import CNNDetector

# Load model
model = CNNDetector()
# ... load checkpoint ...

# Evaluate
evaluator = RobustnessEvaluator(model, device='cuda')
results = evaluator.evaluate_comprehensive(test_loader)
```

---

## 📊 Adversarial Attacks

### Implemented Attacks

1. **FGSM** (Fast Gradient Sign Method)
   - Fast single-step attack
   - Epsilon parameter controls perturbation magnitude

2. **PGD** (Projected Gradient Descent)
   - Iterative version of FGSM
   - Stronger and more effective

3. **C&W** (Carlini & Wagner)
   - Optimization-based attack
   - Minimizes L2 perturbation

4. **DeepFool**
   - Finds minimal perturbation to decision boundary
   - Geometry-based approach

5. **Spectral Perturbations**
   - Audio-specific attacks
   - Time warping, frequency masking

### Example Usage

```python
from src.attacks import FGSM, PGD

# FGSM attack
fgsm = FGSM(model=detector, epsilon=0.03)
adv_examples, metrics = fgsm.generate(inputs, labels)

# PGD attack
pgd = PGD(model=detector, epsilon=0.03, alpha=0.01, num_iter=10)
adv_examples, metrics = pgd.generate(inputs, labels)
```

---

## 🛡️ Adversarial Training

Train robust models using adversarial examples:

```python
from src.training.adv_train import AdversarialTrainer

trainer = AdversarialTrainer(
    model=model,
    optimizer=optimizer,
    adv_ratio=0.5,  # 50% adversarial examples
    attack_type='pgd',
    attack_params={'epsilon': 0.03, 'alpha': 0.01, 'num_iter': 7}
)

history = trainer.train(train_loader, val_loader, num_epochs=50)
```

---

## 📦 Model Export

Export models to multiple formats:

```python
from src.export.export_utils import ModelExporter

exporter = ModelExporter(save_dir='artifacts/models', model_name='CNN_Detector')

# Export to all formats
export_paths = exporter.export_all(
    model=model,
    example_input=example_tensor,
    optimizer=optimizer,
    metrics=metrics
)

# Exports:
# - model.pt (PyTorch checkpoint)
# - model_frozen.pt (Frozen model)
# - model.ts (TorchScript)
# - model.onnx (ONNX)
```

---

## 🎯 Ensemble Detection

Combine multiple models for robust detection:

```python
from src.models.ensemble.ensemble import EnsembleDetector

ensemble = EnsembleDetector(
    models=[cnn_detector, rnn_detector, transformer_detector],
    weights=[0.4, 0.3, 0.3],
    voting='soft'
)

# Predict with explanation
result = ensemble.predict_with_explanation(input_audio)
print(result['prediction'])
print(result['confidence'])
print(result['explanation'])
```

---

## 📈 Evaluation and Metrics

Comprehensive robustness evaluation:

- **Clean Accuracy**: Performance on unmodified samples
- **Adversarial Accuracy**: Performance on adversarial examples
- **Robustness Drop**: Difference between clean and adversarial accuracy
- **AUROC, Precision, Recall, F1**
- **Confusion Matrices**
- **ROC Curves**

All metrics are automatically saved to:
- `metrics.json`
- `metrics.csv`
- Visualization plots

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{detectvoice_adversarial,
  title={DetectVoice Adversarial Suite: Robust Audio Deepfake Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/detectvoice_adversarial}
}
```

---

## 📄 License

This project is released under the MIT License with additional ethical use clauses.

**IMPORTANT**: Users must comply with all legal and ethical guidelines. Misuse for creating deepfakes or unauthorized voice cloning is strictly prohibited.

---

## 🤝 Contributing

Contributions for defensive research are welcome! Please:

1. Ensure contributions align with defensive purposes
2. Include tests for new features
3. Follow the code style
4. Update documentation

---

## 📞 Contact

For questions about responsible use or research collaborations:
- Email: your.email@example.com
- Issues: GitHub Issues

---

## 🔒 Security Policy

If you discover a security vulnerability or potential misuse:

1. **DO NOT** open a public issue
2. Email directly to: security@example.com
3. Provide detailed information
4. Allow reasonable time for response

---

## ⚖️ Ethical Guidelines Summary

1. **Defensive Research Only**: This tool is for detection, not creation
2. **Consent Required**: Obtain consent for processing voice data
3. **Legal Compliance**: Follow all applicable laws
4. **No Malicious Use**: Never use for unauthorized impersonation
5. **Transparency**: Disclose limitations and capabilities honestly

**Remember**: With great power comes great responsibility. Use this tool ethically and legally.

---
## 📚 Datasets Utilizados (Voz Real, Sintética y Deepfake)

Este proyecto utiliza y recomienda múltiples datasets de voz real y sintetizada para entrenar y evaluar modelos de detección de deepfakes. Aquí están los enlaces oficiales y las referencias mencionadas previamente:

---

### 🔹 ASVspoof 2019 — Real + Sintética (TTS, VC, Replay)
Dataset clásico y base de investigación para detección de audio falsificado.

- Sitio oficial: https://www.asvspoof.org/index2019.html
- Info completa: https://cisaad.umbc.edu/asvspoof-2019-a-large-scale-public-database-of-synthesized-converted-and-replayed-speech/
- MOS / Listening tests: https://zenodo.org/records/8412617

---

### 🔹 DSD-Corpus — Diverse Synthesizer for Deepfake Voice Detection
Dataset moderno de voces reales y sintetizadas por múltiples TTS.

- Descarga: https://zenodo.org/records/13788455

---

### 🔹 LibriTTS — Voces Reales (Multi-Speaker)
Corpus ampliamente usado para TTS y como base de voces reales.

- LibriTTS (SLR60): https://us.openslr.org/60/
- LibriTTS-R (restaurado – SLR141): https://www.openslr.org/141/
- Información extra: https://korshakov.com/datasets/libritts-r

---

### 🔹 AUDETER — Deepfake Audio Detection Dataset (Alta Realidad)
Dataset de gran escala con sintetizadores modernos (2024–2025).

- Paper: https://arxiv.org/abs/2509.04345

---

### 🔹 Recursos Adicionales (Reales + Sintéticos)
Estos datasets también fueron mencionados previamente y son útiles para robustez y testing:

#### FoR: Fake-or-Real Speech Dataset
- Info: https://cisaad.umbc.edu/for-fake-or-real-dataset-for-synthetic-speech-detection/

#### DEEP-VOICE Dataset (Voice Conversion / Deepfake Recognition)
- HuggingFace: https://huggingface.co/datasets/DynamicSuperb/DeepFakeVoiceRecognition_DEEP-VOICE

#### Half-Truth HAD Dataset (Audio parcialmente sintetizado)
- Paper / Info: https://arxiv.org/abs/2104.03617

#### ELAD-SVDSR Dataset (Long recordings + deepfake)
- Info: https://arxiv.org/abs/2510.00218

---

### ✔ Recomendación de uso
Para máxima robustez del sistema *DetectVoice*, utilizar combinaciones de:

- **Real:** LibriTTS, LibriTTS-R  
- **Sintética:** DSD-Corpus, FoR, DEEP-VOICE  
- **Adversarial / Deepfake:** ASVspoof 2019/2021, AUDETER, HAD  
- **Larga duración:** ELAD-SVDSR  

Estas bases cubren **TTS de baja calidad → alta calidad**, **Voice Conversion**, **deepfakes de última generación**, **audios adversariales**, y **grabaciones reales**.


---
© 2025 DetectVoice Adversarial Suite. All rights reserved.
