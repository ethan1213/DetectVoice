"""Harmonic CNN for fake audio detection (exploits harmonic structure)"""
import torch
import torch.nn as nn

class HarmonicCNN(nn.Module):
    def __init__(self, in_channels=1, harmonic_layers=[64,128,256], temporal_layers=[256,512], num_harmonics=8, num_classes=2, dropout=0.5):
        super().__init__()

        # Harmonic feature extraction (frequency domain)
        harm_layers = []
        prev = in_channels
        for ch in harmonic_layers:
            harm_layers.extend([
                nn.Conv2d(prev, ch, (num_harmonics, 3), padding=(0, 1)),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d((1, 2)),
                nn.Dropout2d(dropout)
            ])
            prev = ch
        self.harmonic_net = nn.Sequential(*harm_layers)

        # Temporal feature extraction
        temp_layers = []
        prev = harmonic_layers[-1]
        for ch in temporal_layers:
            temp_layers.extend([
                nn.Conv1d(prev, ch, 3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            prev = ch
        self.temporal_net = nn.Sequential(*temp_layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(temporal_layers[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, 1, freq, time)
        x = self.harmonic_net(x)  # Detect harmonic patterns
        x = x.squeeze(2) if x.size(2) == 1 else x.mean(dim=2)  # (batch, ch, time)
        x = self.temporal_net(x)
        return self.classifier(x)
