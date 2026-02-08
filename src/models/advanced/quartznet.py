"""QuartzNet: Fast ASR model, adapted for deepfake detection"""
import torch
import torch.nn as nn

class QuartzBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, repeat=5):
        super().__init__()
        layers = []
        for i in range(repeat):
            layers.extend([
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class QuartzNetDetector(nn.Module):
    def __init__(self, in_channels=80, num_classes=2, dropout=0.2):
        super().__init__()
        # Simplified QuartzNet
        self.blocks = nn.Sequential(
            QuartzBlock(in_channels, 256, 33, 5),
            QuartzBlock(256, 256, 39, 5),
            QuartzBlock(256, 512, 51, 5),
            QuartzBlock(512, 512, 63, 5),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)
