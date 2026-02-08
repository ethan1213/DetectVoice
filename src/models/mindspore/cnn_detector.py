"""
MindSpore CNN Model for Audio Detection
"""
import mindspore.nn as nn
from mindspore.ops import operations as P

class CNNMindSpore(nn.Cell):
    """
    2D CNN for spectrograms, implemented in MindSpore.
    This architecture mirrors the PyTorch CNN2D model for consistency.
    """
    def __init__(
        self, 
        in_channels=1, 
        channels=[64, 128, 256], 
        kernel_size=3,
        num_classes=2, 
        dropout=0.5
    ):
        super(CNNMindSpore, self).__init__()
        
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=kernel_size, pad_mode='pad', padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=float(dropout)) # MindSpore expects float
            ])
            prev_ch = ch
            
        self.features = nn.SequentialCell(layers)
        
        # Classifier
        self.pool = P.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        # The input size to the linear layer depends on the output of the feature extractor.
        # Here, we use a placeholder size. In a real scenario, this might need to be
        # calculated based on the input spectrogram dimensions.
        self.fc1 = nn.Dense(channels[-1], 256)
        self.fc1_relu = nn.ReLU()
        self.fc1_dropout = nn.Dropout(p=float(dropout))
        self.fc2 = nn.Dense(256, num_classes)

    def construct(self, x):
        """
        Defines the forward pass of the model.
        In MindSpore, this method is named 'construct'.
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc1_relu(x)
        x = self.fc1_dropout(x)
        x = self.fc2(x)
        return x

