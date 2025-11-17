"""
Audio Spectrogram Transformer (AST) for Deepfake Detection

AST applies Vision Transformer (ViT) architecture to audio spectrograms.
It treats spectrograms as images and applies patch-based transformers.

Reference:
    Gong et al. (2021) "AST: Audio Spectrogram Transformer"
"""

import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig, ASTFeatureExtractor
from typing import Optional, Dict, Tuple
import warnings
import torchaudio


class ASTDetector(nn.Module):
    """
    Audio Spectrogram Transformer for deepfake detection.

    Processes mel spectrograms using transformer architecture
    with patch embeddings (similar to Vision Transformer).
    """

    def __init__(
        self,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_classes: int = 2,
        freeze_base: bool = True,
        freeze_layers: int = 0,
        hidden_dropout: float = 0.1,
        classifier_hidden: int = 256,
        pooling_mode: str = "cls",  # 'cls' or 'mean'
        sample_rate: int = 16000,
        n_mels: int = 128,
    ):
        """
        Initialize AST detector.

        Args:
            pretrained_model: HuggingFace model identifier
            num_classes: Number of output classes
            freeze_base: Freeze base AST model
            freeze_layers: Number of layers to freeze
            hidden_dropout: Dropout rate
            classifier_hidden: Hidden dimension for classifier
            pooling_mode: 'cls' (use CLS token) or 'mean' (mean pooling)
            sample_rate: Audio sample rate
            n_mels: Number of mel bands
        """
        super(ASTDetector, self).__init__()

        self.num_classes = num_classes
        self.pooling_mode = pooling_mode
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Load pre-trained AST
        try:
            self.ast = ASTModel.from_pretrained(pretrained_model)

            # Also load feature extractor for spectrogram computation
            self.feature_extractor = ASTFeatureExtractor.from_pretrained(
                pretrained_model
            )
        except Exception as e:
            warnings.warn(
                f"Could not load pretrained AST model {pretrained_model}. "
                f"Using random initialization. Error: {e}"
            )
            config = ASTConfig()
            self.ast = ASTModel(config)
            self.feature_extractor = None

        # Get hidden size
        self.hidden_size = self.ast.config.hidden_size

        # Freeze base model if specified
        if freeze_base:
            for param in self.ast.parameters():
                param.requires_grad = False

        # Selective layer freezing
        if freeze_layers > 0 and not freeze_base:
            self._freeze_layers(freeze_layers)

        # Create mel spectrogram transform (backup if feature_extractor not available)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels,
            f_min=0,
            f_max=sample_rate // 2,
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.LayerNorm(classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden // 2, num_classes)
        )

    def _freeze_layers(self, num_layers: int):
        """Freeze first N encoder layers."""
        for i in range(num_layers):
            if i < len(self.ast.encoder.layer):
                for param in self.ast.encoder.layer[i].parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_base_model(self):
        """Freeze entire AST model."""
        for param in self.ast.parameters():
            param.requires_grad = False

    def preprocess_audio(
        self,
        waveform: torch.Tensor,
        use_feature_extractor: bool = True
    ) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram.

        Args:
            waveform: Audio waveform (batch, samples)
            use_feature_extractor: Use HuggingFace feature extractor

        Returns:
            Mel spectrogram ready for AST
        """
        if use_feature_extractor and self.feature_extractor is not None:
            # Use HuggingFace feature extractor
            # This handles normalization and expected input format
            inputs = self.feature_extractor(
                waveform.cpu().numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            return inputs.input_values.to(waveform.device)
        else:
            # Fallback: manual spectrogram computation
            # Ensure waveform has correct shape
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)

            # Compute mel spectrogram
            mel_spec = self.mel_transform(waveform)
            mel_spec_db = self.amplitude_to_db(mel_spec)

            # AST expects specific input shape (batch, time, freq)
            # Transpose if needed
            if mel_spec_db.shape[1] != mel_spec_db.shape[2]:
                mel_spec_db = mel_spec_db.transpose(1, 2)

            return mel_spec_db

    def forward(
        self,
        input_values: torch.Tensor,
        is_spectrogram: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_values: Either raw waveform or spectrogram
            is_spectrogram: Whether input is already a spectrogram

        Returns:
            Logits (batch_size, num_classes)
        """
        # Convert to spectrogram if needed
        if not is_spectrogram:
            input_values = self.preprocess_audio(input_values)

        # Pass through AST
        outputs = self.ast(input_values)

        # Get pooled output
        if self.pooling_mode == "cls":
            # Use CLS token (first token)
            pooled = outputs.last_hidden_state[:, 0]
        elif self.pooling_mode == "mean":
            # Mean pooling over all patches
            pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            pooled = outputs.last_hidden_state[:, 0]

        # Classification
        logits = self.classifier(pooled)

        return logits

    def extract_embeddings(
        self,
        input_values: torch.Tensor,
        is_spectrogram: bool = False
    ) -> torch.Tensor:
        """
        Extract AST embeddings.

        Args:
            input_values: Raw waveform or spectrogram
            is_spectrogram: Whether input is already a spectrogram

        Returns:
            Embeddings (batch_size, hidden_size)
        """
        with torch.no_grad():
            if not is_spectrogram:
                input_values = self.preprocess_audio(input_values)

            outputs = self.ast(input_values)

            if self.pooling_mode == "cls":
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }


class CustomAST(nn.Module):
    """
    Custom Audio Spectrogram Transformer implementation.

    Implements AST from scratch without relying on HuggingFace.
    Useful for full customization.
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_fdim: int = 128,  # Frequency dimension
        input_tdim: int = 100,  # Time dimension
        patch_size: Tuple[int, int] = (16, 16),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """
        Initialize custom AST.

        Args:
            num_classes: Number of output classes
            input_fdim: Input frequency dimension
            input_tdim: Input time dimension
            patch_size: Patch size (height, width)
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
        """
        super(CustomAST, self).__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Calculate number of patches
        self.num_patches_f = input_fdim // patch_size[0]
        self.num_patches_t = input_tdim // patch_size[1]
        self.num_patches = self.num_patches_f * self.num_patches_t

        # Patch embedding (convolutional)
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input spectrogram (batch, 1, freq, time)

        Returns:
            Logits (batch, num_classes)
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch, embed_dim, num_patches_f, num_patches_t)
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, num_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classify using CLS token
        x = self.norm(x[:, 0])
        logits = self.head(x)

        return logits
