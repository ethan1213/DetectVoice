"""Conformer: Convolution-augmented Transformer for ASR"""
import torch
import torch.nn as nn

class ConformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, ff_expansion=4, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(nn.Linear(dim, dim*ff_expansion), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim*ff_expansion, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel//2, groups=dim), nn.BatchNorm1d(dim), nn.SiLU())
        self.ffn2 = nn.Sequential(nn.Linear(dim, dim*ff_expansion), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim*ff_expansion, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(self.norm(x))
        x = x + self.attn(self.norm(x), self.norm(x), self.norm(x))[0]
        x_conv = x.transpose(1, 2)
        x = x + self.conv(x_conv).transpose(1, 2)
        x = x + 0.5 * self.ffn2(self.norm(x))
        return x

class ConformerDetector(nn.Module):
    def __init__(self, input_dim=80, encoder_dim=256, num_layers=4, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, encoder_dim)
        self.layers = nn.ModuleList([ConformerBlock(encoder_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        # x: (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
