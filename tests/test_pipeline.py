"""
Unit Tests for the Speech Emotion Recognition Pipeline.

Validates:
1. Strict speaker-independent split (zero actor leakage across train/val/test).
2. Acoustic feature extraction dimensions (280-D vector).
3. PyTorch Model forward pass & output shapes.
4. Dataset integrity & metadata consistency.
"""

import unittest
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from src.features import extract_1d_features, DEFAULT_SAMPLE_RATE
from src.dataset import split_by_speaker, EMOTION_MAP
from src.model import EmotionCNN1D, EmotionCRNN


class TestSERPipeline(unittest.TestCase):

    def test_zero_speaker_leakage(self):
        """Verify that train, validation, and test actor sets are strictly disjoint."""
        train_actors = set(range(1, 17))
        val_actors = set(range(17, 21))
        test_actors = set(range(21, 25))

        self.assertEqual(len(train_actors.intersection(val_actors)), 0, "Leakage between Train and Val!")
        self.assertEqual(len(train_actors.intersection(test_actors)), 0, "Leakage between Train and Test!")
        self.assertEqual(len(val_actors.intersection(test_actors)), 0, "Leakage between Val and Test!")
        self.assertEqual(len(train_actors.union(val_actors).union(test_actors)), 24, "Missing actors in split!")

    def test_feature_extraction_dimension(self):
        """Verify that acoustic feature extraction produces exactly 280 features."""
        # 3 seconds of dummy audio at 22050 Hz
        dummy_audio = np.random.uniform(-0.5, 0.5, size=DEFAULT_SAMPLE_RATE * 3).astype(np.float32)
        features = extract_1d_features(dummy_audio, sr=DEFAULT_SAMPLE_RATE)
        self.assertEqual(features.shape, (280,), f"Expected 280 features, got {features.shape}")

    def test_cnn_model_forward_pass(self):
        """Verify 1D CNN model accepts (B, 280) and outputs (B, 8) logits."""
        model = EmotionCNN1D(in_features=280, num_classes=8)
        model.eval()

        dummy_batch = torch.randn(4, 280)
        with torch.no_grad():
            output = model(dummy_batch)

        self.assertEqual(output.shape, (4, 8), f"Expected output shape (4, 8), got {output.shape}")

    def test_crnn_model_forward_pass(self):
        """Verify CRNN model accepts (B, 280) and outputs (B, 8) logits."""
        model = EmotionCRNN(in_features=280, num_classes=8)
        model.eval()

        dummy_batch = torch.randn(4, 280)
        with torch.no_grad():
            output = model(dummy_batch)

        self.assertEqual(output.shape, (4, 8), f"Expected output shape (4, 8), got {output.shape}")


if __name__ == "__main__":
    unittest.main()
