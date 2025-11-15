"""
Ejemplo de uso completo de DetectVoice v2.
Muestra cómo usar todas las nuevas funcionalidades.
"""

import torch
from pathlib import Path

# Importar modelo
from src.models.cnn_detector import CNNDetector

# Importar utilidades
from src.utils.metrics import MetricsLogger
from src.utils.model_export import freeze_model, export_all_formats

# Importar adversarial (OPCIONAL)
from src.adversarial.fgsm import FGSM
from src.adversarial.pgd import PGD
from src.adversarial.adversarial_evaluator import AdversarialEvaluator


def example_training():
    """Ejemplo de entrenamiento con métricas automáticas."""
    print("="*60)
    print("EJEMPLO 1: Entrenamiento con Métricas")
    print("="*60)

    # 1. Crear modelo
    model = CNNDetector(input_channels=1, num_classes=2, dropout=0.5)
    print(f"✓ Modelo creado: {sum(p.numel() for p in model.parameters()):,} parámetros")

    # 2. Inicializar logger de métricas
    metrics_logger = MetricsLogger(
        model_name="CNN_example",
        save_dir="reports/plots"
    )
    print("✓ MetricsLogger inicializado")

    # 3. Simular entrenamiento (reemplaza con tu código real)
    for epoch in range(1, 6):
        # Tu código de entrenamiento aquí...
        train_loss = 0.5 - (epoch * 0.05)  # Ejemplo
        val_loss = 0.6 - (epoch * 0.04)    # Ejemplo
        train_acc = 0.6 + (epoch * 0.05)   # Ejemplo
        val_acc = 0.55 + (epoch * 0.04)    # Ejemplo

        # Log metrics
        metrics_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 4. Generar todos los gráficos (después del entrenamiento)
    import numpy as np
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)

    metrics_logger.generate_all_plots(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores
    )
    print("\n✓ Todos los gráficos generados en reports/plots/CNN_example/plots/")


def example_export():
    """Ejemplo de exportación de modelo a todos los formatos."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Exportación de Modelo")
    print("="*60)

    # 1. Crear modelo
    model = CNNDetector()
    print("✓ Modelo creado")

    # 2. Congelar modelo
    model = freeze_model(model)
    print("✓ Modelo congelado (requires_grad=False)")

    # 3. Preparar input de ejemplo
    example_input = torch.randn(1, 128, 94)  # (batch, freq, time)
    print(f"✓ Example input shape: {example_input.shape}")

    # 4. Exportar a todos los formatos
    paths = export_all_formats(
        model=model,
        base_path=Path("checkpoints"),
        model_name="CNN_frozen",
        example_input=example_input
    )

    print("\n✓ Modelos exportados:")
    for format_name, path in paths.items():
        print(f"  - {format_name}: {path}")


def example_adversarial():
    """Ejemplo de uso del módulo adversarial OPCIONAL."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Evaluación Adversarial (OPCIONAL)")
    print("="*60)

    # 1. Crear modelo
    model = CNNDetector()
    model.eval()
    print("✓ Modelo creado")

    # 2. Datos de ejemplo
    inputs = torch.randn(4, 128, 94)
    labels = torch.randint(0, 2, (4,))
    print(f"✓ Datos de ejemplo: {inputs.shape}")

    # 3. Ataque FGSM
    print("\n[Ataque FGSM]")
    fgsm = FGSM(model=model, epsilon=0.03, device='cpu')
    adv_inputs, metrics = fgsm.generate(inputs, labels)
    print(f"  L2 norm: {metrics['l2_norm']:.4f}")
    print(f"  L-inf norm: {metrics['linf_norm']:.4f}")

    # 4. Ataque PGD
    print("\n[Ataque PGD]")
    pgd = PGD(model=model, epsilon=0.03, alpha=0.01, num_iter=10, device='cpu')
    adv_inputs, metrics = pgd.generate(inputs, labels)
    print(f"  L2 norm: {metrics['l2_norm']:.4f}")
    print(f"  Iteraciones: {metrics['num_iter']}")

    print("\n✓ Ataques adversariales ejecutados correctamente")


def example_full_adversarial_evaluation():
    """Ejemplo de evaluación adversarial completa."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Evaluación Adversarial Completa")
    print("="*60)

    # 1. Crear modelo
    model = CNNDetector()
    model.eval()

    # 2. Crear dataloader de ejemplo (reemplaza con tu dataloader real)
    from torch.utils.data import TensorDataset, DataLoader

    dummy_inputs = torch.randn(32, 128, 94)
    dummy_labels = torch.randint(0, 2, (32,))
    dataset = TensorDataset(dummy_inputs, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=8)

    # 3. Crear evaluador adversarial
    evaluator = AdversarialEvaluator(
        model=model,
        model_name="CNN_example",
        device='cpu',
        save_dir="reports/adversarial"
    )

    # 4. Evaluar contra todos los ataques
    results = evaluator.evaluate_all_attacks(dataloader)

    print("\n✓ Evaluación completada")
    print(f"✓ Resultados guardados en: reports/adversarial/CNN_example/")


def example_simple_usage():
    """Ejemplo simple de uso diario."""
    print("\n" + "="*60)
    print("EJEMPLO 5: Uso Simple (Integración con tu código)")
    print("="*60)

    # Tu código de entrenamiento existente
    model = CNNDetector()
    # ... entrenar modelo ...

    # Al final del entrenamiento:

    # Opción 1: Solo guardar modelo
    model.save(
        save_path="checkpoints/pt/my_model.pt",
        epoch=50,
        metrics={'val_acc': 0.95}
    )
    print("✓ Modelo guardado")

    # Opción 2: Exportar a todos los formatos
    example_input = torch.randn(1, 128, 94)
    model.export_all(
        base_path="checkpoints",
        model_name="my_model",
        example_input=example_input
    )
    print("✓ Modelo exportado a .pt, .ts, .onnx")

    # Opción 3: Evaluación adversarial (opcional)
    # evaluator = AdversarialEvaluator(model, "my_model")
    # results = evaluator.evaluate_all_attacks(test_loader)
    print("✓ Listo!")


if __name__ == "__main__":
    print("\n" + "🚀 DetectVoice v2 - Ejemplos de Uso\n")

    # Ejecutar ejemplos
    example_training()
    example_export()
    example_adversarial()
    # example_full_adversarial_evaluation()  # Requiere más tiempo
    example_simple_usage()

    print("\n" + "="*60)
    print("✓ Todos los ejemplos completados exitosamente")
    print("="*60)
    print("\nPróximos pasos:")
    print("1. Revisa los gráficos en reports/plots/")
    print("2. Revisa los modelos exportados en checkpoints/")
    print("3. Integra estas funciones en tu código de entrenamiento")
    print("4. (Opcional) Usa el módulo adversarial para testing")
    print("\n🔒 Recuerda: Uso ético y responsable únicamente.")
