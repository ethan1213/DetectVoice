"""
Model export utilities for DetectVoice.
Supports freezing, PyTorch, TorchScript, and ONNX export.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import onnx
import logging

logger = logging.getLogger(__name__)


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze model parameters (requires_grad = False).

    Args:
        model: PyTorch model

    Returns:
        Frozen model
    """
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    logger.info("Model frozen successfully")

    return model


def save_pytorch(
    model: nn.Module,
    save_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None
) -> None:
    """
    Save PyTorch checkpoint.

    Args:
        model: Model to save
        save_path: Path to save checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, save_path)
    logger.info(f"✓ PyTorch checkpoint saved: {save_path}")


def save_torchscript(
    model: nn.Module,
    save_path: Path,
    example_input: torch.Tensor
) -> None:
    """
    Save model as TorchScript.

    Args:
        model: Model to save
        save_path: Path to save TorchScript model
        example_input: Example input for tracing
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    try:
        traced_model = torch.jit.trace(model, example_input)
        torch.jit.save(traced_model, str(save_path))
        logger.info(f"✓ TorchScript model saved: {save_path}")
    except Exception as e:
        logger.error(f"✗ TorchScript export failed: {e}")
        raise


def save_onnx(
    model: nn.Module,
    save_path: Path,
    example_input: torch.Tensor,
    input_names: list = None,
    output_names: list = None,
    opset_version: int = 12
) -> None:
    """
    Save model as ONNX.

    Args:
        model: Model to save
        save_path: Path to save ONNX model
        example_input: Example input for export
        input_names: Input node names
        output_names: Output node names
        opset_version: ONNX opset version
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']

    try:
        torch.onnx.export(
            model,
            example_input,
            str(save_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names
        )

        # Validate
        onnx_model = onnx.load(str(save_path))
        onnx.checker.check_model(onnx_model)

        logger.info(f"✓ ONNX model saved and validated: {save_path}")
    except Exception as e:
        logger.error(f"✗ ONNX export failed: {e}")
        raise


def export_all_formats(
    model: nn.Module,
    base_path: Path,
    model_name: str,
    example_input: torch.Tensor,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None
) -> dict:
    """
    Export model to all formats (.pt, .ts, .onnx).

    Args:
        model: Model to export
        base_path: Base directory for saving
        model_name: Name of the model
        example_input: Example input tensor
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional)
        metrics: Training metrics (optional)

    Returns:
        Dictionary with paths to saved models
    """
    base_path = Path(base_path)
    paths = {}

    # PyTorch checkpoint
    pt_path = base_path / 'pt' / f"{model_name}.pt"
    save_pytorch(model, pt_path, optimizer, epoch, metrics)
    paths['pytorch'] = pt_path

    # TorchScript
    try:
        ts_path = base_path / 'torchscript' / f"{model_name}.ts"
        save_torchscript(model, ts_path, example_input)
        paths['torchscript'] = ts_path
    except Exception as e:
        logger.warning(f"TorchScript export skipped: {e}")

    # ONNX
    try:
        onnx_path = base_path / 'onnx' / f"{model_name}.onnx"
        save_onnx(model, onnx_path, example_input)
        paths['onnx'] = onnx_path
    except Exception as e:
        logger.warning(f"ONNX export skipped: {e}")

    logger.info(f"✓ Model exported to all available formats")
    return paths
