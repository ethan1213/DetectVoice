"""RNN-based models: LSTM, BiLSTM, GRU, BiGRU"""
import torch
import torch.nn as nn

class LSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1])

class BiLSTMDetector(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)

class GRUDetector(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.classifier(h_n[-1])

class BiGRUDetector(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(h)
