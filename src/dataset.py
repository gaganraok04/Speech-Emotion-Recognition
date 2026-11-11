"""
Dataset and Speaker-Independent Split Module for RAVDESS.

Filename Convention (RAVDESS):
Example: '03-01-05-01-02-01-14.wav'
- Part 0 (03): Modality (01 = full AV, 02 = video-only, 03 = audio-only)
- Part 1 (01): Vocal channel (01 = speech, 02 = song)
- Part 2 (05): Emotion:
    01 = Neutral, 02 = Calm, 03 = Happy, 04 = Sad,
    05 = Angry, 06 = Fearful, 07 = Disgust, 08 = Surprised
- Part 3 (01): Emotional intensity (01 = normal, 02 = strong)
- Part 4 (02): Statement (01 = 'Kids are talking...', 02 = 'Dogs are sitting...')
- Part 5 (01): Repetition (01 = 1st repetition, 02 = 2nd repetition)
- Part 6 (14): Actor (01 to 24: Odd = Male, Even = Female)

SPEAKER-INDEPENDENT SPLIT JUSTIFICATION:
A random train/test split on speech clips leaks identity traits (vocal tract length,
formant baseline, speaker-specific pitch range). The model learns to recognize
*who is speaking* rather than *what emotion is expressed*, yielding inflated
unrealistic scores.

By partitioning the dataset strictly by Actor IDs:
- Train: Actors 01 - 16 (8 Male, 8 Female) ~ 66.7%
- Val:   Actors 17 - 20 (2 Male, 2 Female) ~ 16.7%
- Test:  Actors 21 - 24 (2 Male, 2 Female) ~ 16.7%
We guarantee zero speaker leakage while maintaining perfect gender balance across all splits.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.features import load_and_preprocess_audio, extract_1d_features, extract_2d_mfcc_tensor

EMOTION_MAP = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised"
}

# Emotion labels mapped to 0..7
EMOTION_TO_IDX = {name: i for i, name in enumerate(EMOTION_MAP.values())}
IDX_TO_EMOTION = {i: name for name, i in EMOTION_TO_IDX.items()}


def parse_ravdess_filename(file_path: Path) -> Optional[Dict]:
    """Parse metadata from a RAVDESS .wav filename."""
    stem = file_path.stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    try:
        modality = int(parts[0])
        channel = int(parts[1])
        emotion_code = int(parts[2])
        intensity = int(parts[3])
        statement = int(parts[4])
        repetition = int(parts[5])
        actor_id = int(parts[6])

        if emotion_code not in EMOTION_MAP:
            return None

        gender = "female" if (actor_id % 2 == 0) else "male"

        return {
            "path": str(file_path),
            "filename": file_path.name,
            "actor_id": actor_id,
            "gender": gender,
            "emotion_code": emotion_code,
            "emotion": EMOTION_MAP[emotion_code],
            "label": EMOTION_TO_IDX[EMOTION_MAP[emotion_code]],
            "intensity": "normal" if intensity == 1 else "strong",
            "statement": statement,
            "repetition": repetition
        }
    except ValueError:
        return None


def build_metadata_dataframe(data_dir: Path) -> pd.DataFrame:
    """Find all .wav files in data_dir and extract their metadata into a DataFrame."""
    records = []
    for p in data_dir.glob("**/*.wav"):
        meta = parse_ravdess_filename(p)
        if meta:
            records.append(meta)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No valid RAVDESS audio files found in {data_dir}")
    return df


def split_by_speaker(
    df: pd.DataFrame,
    train_actors: Optional[List[int]] = None,
    val_actors: Optional[List[int]] = None,
    test_actors: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into Train, Val, and Test subsets based strictly on speaker Actor IDs.
    Default:
      Train: Actors 01 to 16 (8 Male, 8 Female)
      Val:   Actors 17 to 20 (2 Male, 2 Female)
      Test:  Actors 21 to 24 (2 Male, 2 Female)
    """
    if train_actors is None:
        train_actors = list(range(1, 17))
    if val_actors is None:
        val_actors = list(range(17, 21))
    if test_actors is None:
        test_actors = list(range(21, 25))

    train_df = df[df["actor_id"].isin(train_actors)].reset_index(drop=True)
    val_df = df[df["actor_id"].isin(val_actors)].reset_index(drop=True)
    test_df = df[df["actor_id"].isin(test_actors)].reset_index(drop=True)

    return train_df, val_df, test_df


class SpeechEmotionDataset(Dataset):
    """
    PyTorch Dataset for Speech Emotion Recognition.
    Computes or loads pre-extracted features.
    """
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[pd.DataFrame] = None
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
