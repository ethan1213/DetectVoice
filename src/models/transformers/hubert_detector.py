"""
HuBERT Detector for Audio Deepfake Detection

HuBERT (Hidden-Unit BERT) is a self-supervised speech representation learning
approach. It predicts masked acoustic units, similar to BERT for text.

Reference:
    Hsu et al. (2021) "HuBERT: Self-Supervised Speech Representation Learning
    by Masked Prediction of Hidden Units"
"""

import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig
from typing import Optional, Dict
import warnings


class HuBERTDetector(nn.Module):
    """
    HuBERT-based audio deepfake detector.

    Very similar architecture to Wav2Vec2, but trained with different
    self-supervised objective (masked prediction of hidden units).
    """

    def __init__(
        self,
        pretrained_model: str = "facebook/hubert-base-ls960",
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
        Initialize HuBERT detector.

        Args:
            pretrained_model: HuggingFace model identifier
                Options: 'facebook/hubert-base-ls960', 'facebook/hubert-large-ll60k'
            num_classes: Number of output classes
            freeze_feature_encoder: Freeze CNN feature extractor
            freeze_layers: Number of transformer layers to freeze
            hidden_dropout: Dropout rate
            attention_dropout: Attention dropout rate
            classifier_hidden: Hidden dimension for classifier
            pooling_mode: Pooling strategy ('mean', 'max', 'attention')
            use_weighted_layer_sum: Use weighted sum of all layers
        """
        super(HuBERTDetector, self).__init__()

        self.num_classes = num_classes
        self.pooling_mode = pooling_mode
        self.use_weighted_layer_sum = use_weighted_layer_sum

        # Load pre-trained HuBERT
        try:
            self.hubert = HubertModel.from_pretrained(
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
            config = HubertConfig()
            self.hubert = HubertModel(config)

        # Get hidden size
        self.hidden_size = self.hubert.config.hidden_size

        # Freeze components
        if freeze_feature_encoder:
            self._freeze_feature_encoder()

        if freeze_layers > 0:
            self._freeze_transformer_layers(freeze_layers)

        # Weighted layer sum
        if use_weighted_layer_sum:
            num_layers = self.hubert.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Attention pooling
        if pooling_mode == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
                nn.Softmax(dim=1)
            )

        # Classification head with deeper architecture
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

    def _freeze_feature_encoder(self):
        """Freeze CNN feature encoder."""
        for param in self.hubert.feature_extractor.parameters():
            param.requires_grad = False

        for param in self.hubert.feature_projection.parameters():
            param.requires_grad = False

    def _freeze_transformer_layers(self, num_layers: int):
        """Freeze first N transformer layers."""
        for i in range(num_layers):
            if i < len(self.hubert.encoder.layers):
                for param in self.hubert.encoder.layers[i].parameters():
                    param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True

    def freeze_base_model(self):
        """Freeze entire HuBERT model."""
        for param in self.hubert.parameters():
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
            attention_mask: Attention mask

        Returns:
            Logits (batch_size, num_classes)
        """
        # Pass through HuBERT
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=self.use_weighted_layer_sum
        )

        # Get hidden states
        if self.use_weighted_layer_sum:
            # Weighted sum of all layers
            hidden_states = outputs.hidden_states
            stacked_hidden_states = torch.stack(hidden_states, dim=0)

            # Apply softmax-normalized weights
            norm_weights = torch.nn.functional.softmax(self.layer_weights, dim=0)
            weighted_hidden_states = (stacked_hidden_states * norm_weights.view(-1, 1, 1, 1)).sum(dim=0)

            hidden_states = weighted_hidden_states
        else:
            hidden_states = outputs.last_hidden_state

        # Pool across sequence
        if self.pooling_mode == "mean":
            pooled = hidden_states.mean(dim=1)
        elif self.pooling_mode == "max":
            pooled = hidden_states.max(dim=1)[0]
        elif self.pooling_mode == "attention":
            attention_weights = self.attention_pooling(hidden_states)
            pooled = (hidden_states * attention_weights).sum(dim=1)
        else:
            pooled = hidden_states.mean(dim=1)

        # Classify
        logits = self.classifier(pooled)

        return logits

    def extract_embeddings(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract HuBERT embeddings.

        Args:
            input_values: Raw audio waveform
            attention_mask: Attention mask

        Returns:
            Embeddings (batch_size, hidden_size)
        """
        with torch.no_grad():
            outputs = self.hubert(input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            if self.pooling_mode == "mean":
                embeddings = hidden_states.mean(dim=1)
            elif self.pooling_mode == "max":
                embeddings = hidden_states.max(dim=1)[0]
            else:
                embeddings = hidden_states.mean(dim=1)

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


class HuBERTEnsemble(nn.Module):
    """
    Ensemble of multiple HuBERT models (e.g., base + large).

    Combines predictions from different HuBERT checkpoints.
    """

    def __init__(
        self,
        model_names: list = ["facebook/hubert-base-ls960", "facebook/hubert-large-ll60k"],
        num_classes: int = 2,
        fusion_method: str = "concat",  # 'concat', 'average', 'weighted'
    ):
        """
        Initialize HuBERT ensemble.

        Args:
            model_names: List of HuggingFace model identifiers
            num_classes: Number of output classes
            fusion_method: How to fuse embeddings
        """
        super().__init__()

        self.fusion_method = fusion_method
        self.models = nn.ModuleList()

        total_hidden_size = 0

        for model_name in model_names:
            detector = HuBERTDetector(
                pretrained_model=model_name,
                num_classes=num_classes,
                freeze_base_model=True  # Only train fusion layer
            )
            self.models.append(detector)
            total_hidden_size += detector.hidden_size

        # Fusion layer
        if fusion_method == "concat":
            self.fusion = nn.Linear(total_hidden_size, num_classes)
        elif fusion_method == "weighted":
            self.fusion_weights = nn.Parameter(torch.ones(len(model_names)) / len(model_names))

    def forward(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Forward pass through ensemble."""
        if self.fusion_method == "concat":
            # Concatenate embeddings
            embeddings = []
            for model in self.models:
                emb = model.extract_embeddings(input_values, attention_mask)
                embeddings.append(emb)

            fused = torch.cat(embeddings, dim=1)
            logits = self.fusion(fused)

        elif self.fusion_method == "average":
            # Average logits
            all_logits = []
            for model in self.models:
                logits = model(input_values, attention_mask)
                all_logits.append(logits)

            logits = torch.stack(all_logits, dim=0).mean(dim=0)

        elif self.fusion_method == "weighted":
            # Weighted average of logits
            all_logits = []
            for model in self.models:
                logits = model(input_values, attention_mask)
                all_logits.append(logits)

            stacked_logits = torch.stack(all_logits, dim=0)
            norm_weights = torch.nn.functional.softmax(self.fusion_weights, dim=0)
            logits = (stacked_logits * norm_weights.view(-1, 1, 1)).sum(dim=0)

        return logits
