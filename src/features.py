"""
Audio Feature Extraction Module for Speech Emotion Recognition (SER).

Extracts acoustic features from speech audio files:
1. MFCC (Mel-Frequency Cepstral Coefficients) - 40 bands capturing spectral envelope & timbre.
2. Delta & Delta-Delta MFCCs - 1st and 2nd temporal derivatives representing pitch velocity & acceleration.
3. Chroma STFT - 12 pitch classes capturing harmonic content and musical intonation.
4. Spectral Contrast - measures energy difference between spectral peaks and valleys.
5. RMS Energy - loudness dynamics across frames.

Provides both:
- 1D Aggregated Feature Vectors (mean + std pooling) for fast, lightweight classification.
- 2D Time-Distributed Feature Matrices (channels x frequency x time) for 1D/2D CNN architectures.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Audio loading defaults
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_DURATION = 3.0  # seconds
TARGET_LENGTH = int(DEFAULT_SAMPLE_RATE * DEFAULT_DURATION)
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048


def load_and_preprocess_audio(
    file_path: str,
    target_sr: int = DEFAULT_SAMPLE_RATE,
    duration: float = DEFAULT_DURATION,
    trim_silence: bool = True
) -> np.ndarray:
    """
    Load an audio file, normalize amplitude, optionally trim leading/trailing silence,
    and enforce a constant length via padding or center cropping.
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)

    if trim_silence and len(y) > 0:
        y, _ = librosa.effects.trim(y, top_db=25)

    target_samples = int(target_sr * duration)
    if len(y) < target_samples:
        # Pad with zeros (symmetrically centered)
        pad_width = target_samples - len(y)
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        y = np.pad(y, (pad_left, pad_right), mode='constant')
    elif len(y) > target_samples:
        # Center crop
        start = (len(y) - target_samples) // 2
        y = y[start:start + target_samples]

    # Normalize amplitude to [-1.0, 1.0]
    max_amp = np.max(np.abs(y))
    if max_amp > 1e-6:
        y = y / max_amp

    return y


def extract_1d_features(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = N_MFCC
) -> np.ndarray:
    """
    Extract aggregated statistical summary features (mean & standard deviation)
    across time frames.
    
    Feature vector breakdown:
    - MFCC (40 mean + 40 std)               = 80
    - Delta MFCC (40 mean + 40 std)         = 80
    - Delta-Delta MFCC (40 mean + 40 std)   = 80
    - Chroma STFT (12 mean + 12 std)        = 24
    - Spectral Contrast (7 mean + 7 std)    = 14
    - RMS Energy (1 mean + 1 std)           = 2
    Total dimensions                        = 280
    """
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 2. Delta & Delta-Delta MFCC
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
    mfcc_delta2_std = np.std(mfcc_delta2, axis=1)

    # 3. Chroma STFT (pitch/harmony)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # 4. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # 5. Root Mean Square (RMS) Energy
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    rms_mean = np.mean(rms, axis=1)
    rms_std = np.std(rms, axis=1)

    feature_vector = np.concatenate([
        mfcc_mean, mfcc_std,
        mfcc_delta_mean, mfcc_delta_std,
        mfcc_delta2_mean, mfcc_delta2_std,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std,
        rms_mean, rms_std
    ])
    return feature_vector.astype(np.float32)


def extract_2d_mfcc_tensor(
    y: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    max_frames: int = 130
) -> np.ndarray:
    """
    Extract a multi-channel 2D representation:
    Channel 0: MFCC (40 x T)
    Channel 1: Delta MFCC (40 x T)
    Channel 2: Delta-Delta MFCC (40 x T)
    Output shape: (3, 40, max_frames) for 2D CNN input.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Pad or crop time frames to max_frames
    def fit_frames(mat: np.ndarray) -> np.ndarray:
        if mat.shape[1] < max_frames:
            pad = max_frames - mat.shape[1]
            return np.pad(mat, ((0, 0), (0, pad)), mode='constant')
        return mat[:, :max_frames]

    stack = np.stack([fit_frames(mfcc), fit_frames(delta), fit_frames(delta2)], axis=0)
    return stack.astype(np.float32)
