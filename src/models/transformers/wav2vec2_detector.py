"""
Wav2Vec 2.0 Detector for Audio Deepfake Detection

Wav2Vec 2.0 is a self-supervised learning framework for speech representations.
Pre-trained on large amounts of unlabeled audio, it can be fine-tuned for
deepfake detection with excellent results.

Reference:
    Baevski et al. (2020) "wav2vec 2.0: A Framework for Self-Supervised
    Learning of Speech Representations"
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
from typing import Optional, Dict, Tuple
import warnings


class Wav2Vec2Detector(nn.Module):
    """
    Wav2Vec2-based audio deepfake detector.

    Supports:
    - Full fine-tuning
    - Partial fine-tuning (freeze feature encoder)
    - Selective layer freezing
    - Multiple pooling strategies
    """

    def __init__(
        self,
        pretrained_model: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        freeze_feature_encoder: bool = True,
        freeze_layers: int = 6,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        classifier_hidden: int = 256,
        pooling_mode: str = "mean",
        use_weighted_layer_sum: bool = False,
    ):
        """
        Initialize Wav2Vec2 detector.

        Args:
            pretrained_model: HuggingFace model identifier
            num_classes: Number of output classes (2 for real/fake)
            freeze_feature_encoder: Freeze CNN feature extractor
            freeze_layers: Number of transformer layers to freeze (0 = none)
            hidden_dropout: Dropout rate for hidden states
            attention_dropout: Dropout rate for attention
            classifier_hidden: Hidden dimension for classifier head
            pooling_mode: How to pool transformer outputs ('mean', 'max', 'attention')
            use_weighted_layer_sum: Use weighted sum of all layer outputs
        """
        super(Wav2Vec2Detector, self).__init__()

        self.num_classes = num_classes
        self.pooling_mode = pooling_mode
        self.use_weighted_layer_sum = use_weighted_layer_sum

        # Load pre-trained Wav2Vec2
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(
                pretrained_model,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                output_hidden_states=use_weighted_layer_sum,
            )
        except Exception as e:
            warnings.warn(
                f"Could not load pretrained model {pretrained_model}. "
                f"Using random initialization. Error: {e}"
            )
            config = Wav2Vec2Config()
            self.wav2vec2 = Wav2Vec2Model(config)

        # Get hidden size from model config
        self.hidden_size = self.wav2vec2.config.hidden_size

        # Freeze feature encoder if specified
        if freeze_feature_encoder:
            self._freeze_feature_encoder()

        # Freeze transformer layers if specified
        if freeze_layers > 0:
            self._freeze_transformer_layers(freeze_layers)

        # Weighted layer sum (if enabled)
        if use_weighted_layer_sum:
            num_layers = self.wav2vec2.config.num_hidden_layers + 1  # +1 for input
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Attention pooling (if selected)
        if pooling_mode == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
                nn.Softmax(dim=1)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, classifier_hidden),
            nn.LayerNorm(classifier_hidden),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.LayerNorm(classifier_hidden // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden // 2, num_classes)
        )

    def _freeze_feature_encoder(self):
        """Freeze the CNN feature encoder."""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False

        # Also freeze feature projection
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = False

    def _freeze_transformer_layers(self, num_layers: int):
        """Freeze first N transformer layers."""
        for i in range(num_layers):
            if i < len(self.wav2vec2.encoder.layers):
                for param in self.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_base_model(self):
        """Freeze entire Wav2Vec2 base model (train only classifier)."""
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_values: Raw audio waveform (batch_size, sequence_length)
            attention_mask: Attention mask (batch_size, sequence_length)

        Returns:
            Logits (batch_size, num_classes)
        """
        # Pass through Wav2Vec2
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.use_weighted_layer_sum
        )

        # Get hidden states
        if self.use_weighted_layer_sum:
            # Weighted sum of all layers
            hidden_states = outputs.hidden_states
            stacked_hidden_states = torch.stack(hidden_states, dim=0)  # (num_layers, batch, seq, hidden)

            # Apply layer weights
            norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
            weighted_hidden_states = (stacked_hidden_states * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)

            hidden_states = weighted_hidden_states
        else:
            # Use last hidden state
            hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

        # Pool across sequence dimension
        if self.pooling_mode == "mean":
            pooled = hidden_states.mean(dim=1)  # (batch, hidden)
        elif self.pooling_mode == "max":
            pooled = hidden_states.max(dim=1)[0]  # (batch, hidden)
        elif self.pooling_mode == "attention":
            # Attention-weighted pooling
            attention_weights = self.attention_pooling(hidden_states)  # (batch, seq, 1)
            pooled = (hidden_states * attention_weights).sum(dim=1)  # (batch, hidden)
        else:
            pooled = hidden_states.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)

        return logits

    def extract_embeddings(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract embeddings (before classification head).

        Args:
            input_values: Raw audio waveform
            attention_mask: Attention mask

        Returns:
            Embeddings (batch_size, hidden_size)
        """
        with torch.no_grad():
            outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            # Pool
            if self.pooling_mode == "mean":
                embeddings = hidden_states.mean(dim=1)
            elif self.pooling_mode == "max":
                embeddings = hidden_states.max(dim=1)[0]
            else:
                embeddings = hidden_states.mean(dim=1)

        return embeddings

    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get count of trainable parameters.

        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }


class Wav2Vec2ForSequenceClassification(nn.Module):
    """
    Simpler wrapper using HuggingFace's built-in classification head.

    This is a lightweight alternative to Wav2Vec2Detector.
    """

    def __init__(
        self,
        pretrained_model: str = "facebook/wav2vec2-base",
        num_classes: int = 2,
        freeze_base: bool = True,
    ):
        """
        Initialize sequence classification model.

        Args:
            pretrained_model: HuggingFace model identifier
            num_classes: Number of output classes
            freeze_base: Freeze base Wav2Vec2 model
        """
        super().__init__()

        from transformers import Wav2Vec2ForSequenceClassification as W2V2Classifier

        try:
            self.model = W2V2Classifier.from_pretrained(
                pretrained_model,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        except Exception as e:
            warnings.warn(f"Could not load pretrained model: {e}")
            config = Wav2Vec2Config(num_labels=num_classes)
            self.model = W2V2Classifier(config)

        if freeze_base:
            # Freeze everything except classifier
            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass."""
        outputs = self.model(input_values, attention_mask=attention_mask)
        return outputs.logits
