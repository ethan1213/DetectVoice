"""Hybrid CRNN model"""
import torch
import torch.nn as nn

class CRNNDetector(nn.Module):
    """CNN + RNN hybrid for spectrogram features"""
    def __init__(self, in_channels=1, cnn_channels=[64,128,256], rnn_hidden=256, rnn_layers=2, bidirectional=True, num_classes=2, dropout=0.5):
        super().__init__()

        # CNN layers
        cnn_layers = []
        prev_ch = in_channels
        for ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),  # Pool only in frequency
                nn.Dropout2d(dropout)
            ])
            prev_ch = ch
        self.cnn = nn.Sequential(*cnn_layers)

        # RNN layers
        self.rnn = nn.LSTM(cnn_channels[-1], rnn_hidden, rnn_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if rnn_layers > 1 else 0)

        # Classifier
        rnn_out_size = rnn_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (batch, channels, freq, time)
        x = self.cnn(x)  # (batch, cnn_channels[-1], freq', time)

        # Pool frequency dimension
        x = x.mean(dim=2)  # (batch, cnn_channels[-1], time)
        x = x.transpose(1, 2)  # (batch, time, cnn_channels[-1])

        # RNN
        _, (h_n, _) = self.rnn(x)
        if self.rnn.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]

        return self.classifier(h)
