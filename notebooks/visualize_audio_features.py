"""
Audio Visualization and Exploratory Data Analysis (EDA) Script.

Generates visual comparisons for Speech Emotion Recognition:
1. Raw Audio Waveforms: High-Arousal (Angry) vs Low-Arousal (Sad) vs Neutral.
2. Mel-Spectrograms across emotions.
3. 40-band MFCC representations.
4. Pitch (F0) & RMS Energy distribution comparisons.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = PROJECT_ROOT / "web" / "samples"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating exploratory audio feature visualizations...")

    emotions_to_compare = ["angry", "happy", "sad", "neutral"]
    files = {
        emo: SAMPLES_DIR / f"test_female_{emo}.wav"
        for emo in emotions_to_compare
        if (SAMPLES_DIR / f"test_female_{emo}.wav").exists()
    }

    if len(files) < 2:
        print("Sample files not found in web/samples. Exiting.")
        return

    # 1. Plot Waveforms Comparison
    fig, axes = plt.subplots(len(files), 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Speech Waveform Comparison Across Emotional States", fontsize=14, fontweight="bold")

    audio_signals = {}
    sr = 22050

    for idx, (emo, path) in enumerate(files.items()):
        y, _ = librosa.load(str(path), sr=sr, duration=3.0)
        audio_signals[emo] = y
        ax = axes[idx]
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#3b82f6" if emo in ("sad", "neutral") else "#ef4444")
        ax.set_title(f"Emotion: {emo.upper()}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "waveforms_comparison.png", dpi=300)
    plt.close()
    print("Saved: waveforms_comparison.png")

    # 2. Plot Mel-Spectrograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Mel-Spectrogram Comparison (Log-Power Frequency)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (emo, y) in enumerate(audio_signals.items()):
        ax = axes[idx]
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap="magma")
        ax.set_title(f"Mel-Spectrogram: {emo.upper()}", fontsize=11, fontweight="bold")
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mel_spectrograms_comparison.png", dpi=300)
    plt.close()
    print("Saved: mel_spectrograms_comparison.png")

    # 3. Plot 40-band MFCCs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("40-band Mel-Frequency Cepstral Coefficients (MFCCs)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for idx, (emo, y) in enumerate(audio_signals.items()):
        ax = axes[idx]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        img = librosa.display.specshow(mfcc, x_axis='time', ax=ax, cmap="coolwarm")
        ax.set_title(f"MFCCs (40 bands): {emo.upper()}", fontsize=11, fontweight="bold")
        fig.colorbar(img, ax=ax)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mfcc_comparison.png", dpi=300)
    plt.close()
    print("Saved: mfcc_comparison.png")

    print(f"All visualizations successfully generated in {FIGURES_DIR}!")


if __name__ == "__main__":
    main()
