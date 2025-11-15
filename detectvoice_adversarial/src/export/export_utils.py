"""
Model export utilities for DetectVoice.

Exports models to:
- PyTorch (.pt)
- Frozen PyTorch (.pt with requires_grad=False)
- TorchScript (.ts)
- ONNX (.onnx)

⚠️  SECURITY & ETHICS NOTICE ⚠️
Only export detection models, not generation models.
Exported models should be used for defensive purposes only.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict
import onnx
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelExporter:
    """
    Export PyTorch models to multiple formats.
    """

    def __init__(self, save_dir: Path, model_name: str):
        """
        Initialize model exporter.

        Args:
            save_dir: Directory to save exported models
            model_name: Name of the model
        """
        self.save_dir = Path(save_dir) / model_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        logger.info(f"Model exporter initialized for: {model_name}")
        logger.info(f"Export directory: {self.save_dir}")

    def save_pytorch_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None,
        config: Optional[Dict] = None
    ) -> Path:
        """
        Save full PyTorch checkpoint (.pt).

        Args:
            model: PyTorch model
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
            config: Model configuration (optional)

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_name': self.model_name,
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if metrics is not None:
            checkpoint['metrics'] = metrics

        if config is not None:
            checkpoint['config'] = config

        save_path = self.save_dir / "model.pt"
        torch.save(checkpoint, save_path)

        logger.info(f"✓ PyTorch checkpoint saved: {save_path}")
        return save_path

    def save_frozen_model(self, model: nn.Module) -> Path:
        """
        Save frozen model (.pt with requires_grad=False).

        Args:
            model: PyTorch model

        Returns:
            Path to saved frozen model
        """
        # Freeze all parameters
        model_frozen = model
        for param in model_frozen.parameters():
            param.requires_grad = False

        model_frozen.eval()

        checkpoint = {
            'model_state_dict': model_frozen.state_dict(),
            'model_name': self.model_name,
            'frozen': True
        }

        save_path = self.save_dir / "model_frozen.pt"
        torch.save(checkpoint, save_path)

        logger.info(f"✓ Frozen model saved: {save_path}")
        return save_path

    def save_torchscript(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        use_trace: bool = True
    ) -> Path:
        """
        Save TorchScript model (.ts).

        Args:
            model: PyTorch model
            example_input: Example input tensor for tracing
            use_trace: Use tracing (True) or scripting (False)

        Returns:
            Path to saved TorchScript model
        """
        model.eval()

        try:
            if use_trace:
                logger.info("Tracing model for TorchScript...")
                scripted_model = torch.jit.trace(model, example_input)
            else:
                logger.info("Scripting model for TorchScript...")
                scripted_model = torch.jit.script(model)

            save_path = self.save_dir / "model.ts"
            torch.jit.save(scripted_model, str(save_path))

            logger.info(f"✓ TorchScript model saved: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"✗ TorchScript export failed: {e}")
            raise

    def save_onnx(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        input_names: list = None,
        output_names: list = None,
        dynamic_axes: dict = None,
        opset_version: int = 12
    ) -> Path:
        """
        Save ONNX model (.onnx).

        Args:
            model: PyTorch model
            example_input: Example input tensor
            input_names: Names of input nodes
            output_names: Names of output nodes
            dynamic_axes: Dynamic axes for variable input size
            opset_version: ONNX opset version

        Returns:
            Path to saved ONNX model
        """
        model.eval()

        if input_names is None:
            input_names = ['input']

        if output_names is None:
            output_names = ['output']

        save_path = self.save_dir / "model.onnx"

        try:
            logger.info("Exporting to ONNX...")

            torch.onnx.export(
                model,
                example_input,
                str(save_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )

            # Validate ONNX model
            logger.info("Validating ONNX model...")
            onnx_model = onnx.load(str(save_path))
            onnx.checker.check_model(onnx_model)

            logger.info(f"✓ ONNX model saved and validated: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"✗ ONNX export failed: {e}")
            raise

    def export_all(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """
        Export model to all formats.

        Args:
            model: PyTorch model
            example_input: Example input tensor
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)

        Returns:
            Dictionary with paths to all exported models
        """
        logger.info("=" * 60)
        logger.info(f"Exporting {self.model_name} to all formats...")
        logger.info("=" * 60)

        paths = {}

        # 1. PyTorch checkpoint
        paths['pytorch'] = self.save_pytorch_checkpoint(
            model, optimizer, epoch, metrics
        )

        # 2. Frozen model
        paths['frozen'] = self.save_frozen_model(model)

        # 3. TorchScript
        try:
            paths['torchscript'] = self.save_torchscript(model, example_input)
        except Exception as e:
            logger.warning(f"TorchScript export skipped: {e}")

        # 4. ONNX
        try:
            paths['onnx'] = self.save_onnx(model, example_input)
        except Exception as e:
            logger.warning(f"ONNX export skipped: {e}")

        logger.info("=" * 60)
        logger.info("✓ Export complete!")
        logger.info(f"All models saved to: {self.save_dir}")
        logger.info("=" * 60)

        return paths


def load_pytorch_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: str = 'cpu'
) -> nn.Module:
    """
    Load PyTorch checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: Model instance
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from: {checkpoint_path}")

    return model


def load_torchscript(
    model_path: Path,
    device: str = 'cpu'
) -> torch.jit.ScriptModule:
    """
    Load TorchScript model.

    Args:
        model_path: Path to TorchScript model
        device: Device to load model on

    Returns:
        Loaded TorchScript model
    """
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    logger.info(f"TorchScript model loaded from: {model_path}")

    return model
