"""
PyTorch Neural Network Architectures for Speech Emotion Recognition.

Contains:
1. EmotionCNN1D: Deep 1D Convolutional Network with residual-style blocks,
   batch normalization, adaptive pooling, and dropout.
2. EmotionCRNN: Convolutional Recurrent Network combining CNN feature maps
   with Bidirectional LSTM to capture temporal dynamics in speech prosody.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Convolutional Block with BatchNorm, LeakyReLU, and Dropout."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class EmotionCNN1D(nn.Module):
    """
    1D CNN Classifier for speech features.
    Accepts 1D feature vectors (e.g. 280 dims) reshaped to (B, 1, 280).
    """
    def __init__(self, in_features: int = 280, num_classes: int = 8):
        super().__init__()
        self.block1 = Conv1DBlock(1, 64, kernel_size=5, dropout=0.25)
        self.block2 = Conv1DBlock(64, 128, kernel_size=5, dropout=0.3)
        self.block3 = Conv1DBlock(128, 256, kernel_size=3, dropout=0.35)
        self.block4 = Conv1DBlock(256, 256, kernel_size=3, dropout=0.35)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is 2D (B, features), unsqueeze channel dim: (B, 1, features)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x)
        return logits


class EmotionCRNN(nn.Module):
    """
    Convolutional-Recurrent Neural Network (CRNN).
    Combines 1D CNN feature extractors with a Bidirectional LSTM.
    """
    def __init__(self, in_features: int = 280, num_classes: int = 8, hidden_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        # Sequence length after two poolings of 280 is 70
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)  # (B, 128, T)
        x = x.permute(0, 2, 1)  # (B, T, 128)
        lstm_out, _ = self.lstm(x)  # (B, T, hidden_size * 2)
        # Average pooling over time steps
        out = torch.mean(lstm_out, dim=1)
        logits = self.fc(out)
        return logits


def get_model(model_name: str = "cnn1d", in_features: int = 280, num_classes: int = 8) -> nn.Module:
    """Factory helper to retrieve model by name."""
    if model_name.lower() == "cnn1d":
        return EmotionCNN1D(in_features=in_features, num_classes=num_classes)
    elif model_name.lower() in ("crnn", "lstm"):
        return EmotionCRNN(in_features=in_features, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Choose 'cnn1d' or 'crnn'.")
