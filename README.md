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
