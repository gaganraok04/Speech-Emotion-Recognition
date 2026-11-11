"""
Single Audio Inference Script for Speech Emotion Recognition.

Usage:
  python -m src.predict --audio "path/to/recording.wav"
"""

import argparse
from pathlib import Path
import numpy as np
import torch

from src.features import (
    load_and_preprocess_audio,
    extract_1d_features,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DURATION
)
from src.model import get_model
from src.dataset import IDX_TO_EMOTION

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def predict_audio(audio_path: str, model_name: str = "cnn1d"):
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load precomputed scaler statistics
    features_npz = PROCESSED_DIR / "ravdess_features.npz"
    if not features_npz.exists():
        raise FileNotFoundError("Missing ravdess_features.npz. Preprocessing must be completed first.")

    npz_data = np.load(features_npz)
    scaler_mean = npz_data["scaler_mean"]
    scaler_scale = npz_data["scaler_scale"]

    # Extract acoustic features
    print(f"Loading and analyzing audio: {path.name}")
    y = load_and_preprocess_audio(
        str(path),
        target_sr=DEFAULT_SAMPLE_RATE,
        duration=DEFAULT_DURATION,
        trim_silence=True
    )
    raw_feats = extract_1d_features(y, sr=DEFAULT_SAMPLE_RATE)

    # Standardize
    norm_feats = (raw_feats - scaler_mean) / (scaler_scale + 1e-8)

    # Load PyTorch model
    checkpoint_path = MODELS_DIR / f"best_{model_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = get_model(model_name=model_name, in_features=len(norm_feats), num_classes=8)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Predict
    input_tensor = torch.tensor(norm_feats, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_emotion = IDX_TO_EMOTION[pred_idx]
    pred_confidence = probs[pred_idx] * 100

    print("\n" + "=" * 45)
    print(f"  PREDICTED EMOTION: {pred_emotion.upper()} ({pred_confidence:.1f}% confidence)")
    print("=" * 45)

    print("\nProbability Distribution:")
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        emo = IDX_TO_EMOTION[idx]
        bar = "█" * int(probs[idx] * 30)
        print(f"  {emo.capitalize():<10} | {probs[idx]*100:5.1f}% | {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from speech file")
    parser.add_argument("--audio", type=str, required=True, help="Path to .wav audio file")
    parser.add_argument("--model", type=str, default="cnn1d", choices=["cnn1d", "crnn"])
    args = parser.parse_args()

    predict_audio(args.audio, args.model)
