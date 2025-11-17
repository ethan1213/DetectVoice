"""
ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN
State-of-the-art for speaker verification, excellent for deepfake detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.shape
        y = x.mean(dim=2)
        y = self.fc(y).unsqueeze(2)
        return x * y

class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale features"""
    def __init__(self, channels, kernel_size=3, scale=8):
        super().__init__()
        self.scale = scale
        self.convs = nn.ModuleList([
            nn.Conv1d(channels // scale, channels // scale, kernel_size, padding=kernel_size//2)
            for _ in range(scale - 1)
        ])
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        spx = torch.chunk(x, self.scale, dim=1)
        outputs = [spx[0]]
        for i in range(1, self.scale):
            if i == 1:
                sp = spx[i]
            else:
                sp = spx[i] + outputs[-1]
            sp = self.convs[i-1](sp)
            outputs.append(sp)
        out = torch.cat(outputs, dim=1)
        return F.relu(self.bn(out))

class ECAPATDNNDetector(nn.Module):
    """ECAPA-TDNN for deepfake detection"""
    def __init__(self, in_channels=80, channels=1024, emb_dim=192, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, 5, padding=2)
        self.res2net = Res2NetBlock(channels, 3, scale=8)
        self.se = SEBlock(channels)
        self.conv2 = nn.Conv1d(channels, emb_dim, 1)
        self.bn = nn.BatchNorm1d(emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res2net(x)
        x = self.se(x)
        x = F.relu(self.bn(self.conv2(x)))
        x = x.mean(dim=2)  # Temporal pooling
        return self.classifier(x)
