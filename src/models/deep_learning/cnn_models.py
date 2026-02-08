"""CNN Models for Audio (1D and 2D)"""
import torch
import torch.nn as nn

class CNN1D(nn.Module):
    """1D CNN for raw waveform"""
    def __init__(self, in_channels=1, channels=[64,128,256,512], num_classes=2, dropout=0.5):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv1d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout)
            ])
            prev_ch = ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class CNN2D(nn.Module):
    """2D CNN for spectrograms"""
    def __init__(self, in_channels=1, channels=[64,128,256,512], num_classes=2, dropout=0.5):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout)
            ])
            prev_ch = ch
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
