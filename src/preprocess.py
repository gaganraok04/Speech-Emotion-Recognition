"""
Preprocess RAVDESS Dataset and Cache Extracted Features.

Steps:
1. Parse all 1,440 RAVDESS audio files.
2. Perform Speaker-Independent partitioning:
   - Train: Actors 01 to 16 (8 Male, 8 Female)
   - Val:   Actors 17 to 20 (2 Male, 2 Female)
   - Test:  Actors 21 to 24 (2 Male, 2 Female)
3. Extract acoustic feature representations (MFCCs, Deltas, Delta-Deltas, Chroma, Spectral Contrast, RMS).
4. Save precomputed feature arrays to data/processed/.
5. Select sample audio clips from test speakers for the Web UI demo.
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from src.dataset import (
    build_metadata_dataframe,
    split_by_speaker,
    EMOTION_MAP,
    EMOTION_TO_IDX
)
from src.features import (
    load_and_preprocess_audio,
    extract_1d_features,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DURATION
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLES_DIR = PROJECT_ROOT / "web" / "samples"


def extract_features_for_df(df: pd.DataFrame, desc: str = "Extracting") -> Tuple[np.ndarray, np.ndarray]:
    """Extract 1D feature vectors for all audio files in the dataframe."""
    features_list = []
    labels_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        audio_path = row["path"]
        label = row["label"]
        try:
            y = load_and_preprocess_audio(
                audio_path,
                target_sr=DEFAULT_SAMPLE_RATE,
                duration=DEFAULT_DURATION,
                trim_silence=True
            )
            feat = extract_1d_features(y, sr=DEFAULT_SAMPLE_RATE)
            features_list.append(feat)
            labels_list.append(label)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    return np.array(features_list, dtype=np.float32), np.array(labels_list, dtype=np.int64)


def copy_web_test_samples(test_df: pd.DataFrame, target_dir: Path):
    """
    Select representative audio clips from test actors (21-24)
    covering each emotion, to bundle with the static Web UI.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    sample_records = []

    # Pick one male sample and one female sample per emotion from test actors
    for emotion_idx, emotion_name in enumerate(EMOTION_MAP.values()):
        emo_df = test_df[test_df["emotion"] == emotion_name]
        
        # Male sample (odd actor ID)
        male_samples = emo_df[emo_df["actor_id"] % 2 != 0]
        if not male_samples.empty:
            m_row = male_samples.iloc[0]
            m_src = Path(m_row["path"])
            m_dst_name = f"test_male_{emotion_name}.wav"
            shutil.copy2(m_src, target_dir / m_dst_name)
            sample_records.append({
                "emotion": emotion_name,
                "label": emotion_idx,
                "gender": "male",
                "actor_id": int(m_row["actor_id"]),
                "filename": m_dst_name,
                "original_file": m_src.name
            })

        # Female sample (even actor ID)
        female_samples = emo_df[emo_df["actor_id"] % 2 == 0]
        if not female_samples.empty:
            f_row = female_samples.iloc[0]
            f_src = Path(f_row["path"])
            f_dst_name = f"test_female_{emotion_name}.wav"
            shutil.copy2(f_src, target_dir / f_dst_name)
            sample_records.append({
                "emotion": emotion_name,
                "label": emotion_idx,
                "gender": "female",
                "actor_id": int(f_row["actor_id"]),
                "filename": f_dst_name,
                "original_file": f_src.name
            })

    samples_meta_df = pd.DataFrame(sample_records)
    samples_meta_df.to_json(target_dir / "samples.json", orient="records", indent=2)
    print(f"Copied {len(sample_records)} test sample audio clips to {target_dir}")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing RAVDESS files...")
    df = build_metadata_dataframe(RAW_DIR)
    print(f"Total audio files found: {len(df)}")
    print(f"Actors found: {sorted(df['actor_id'].unique().tolist())}")

    # Speaker-Independent Split
    train_df, val_df, test_df = split_by_speaker(df)
    print("\nSpeaker-Independent Split Summary:")
    print(f"  Train Set: {len(train_df)} clips (Actors 1..16, {train_df['gender'].value_counts().to_dict()})")
    print(f"  Val Set:   {len(val_df)} clips (Actors 17..20, {val_df['gender'].value_counts().to_dict()})")
    print(f"  Test Set:  {len(test_df)} clips (Actors 21..24, {test_df['gender'].value_counts().to_dict()})")

    # Save metadata splits
    train_df.to_csv(PROCESSED_DIR / "train_meta.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val_meta.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_meta.csv", index=False)

    print("\nExtracting features for Train set...")
    X_train, y_train = extract_features_for_df(train_df, desc="Train features")

    print("\nExtracting features for Val set...")
    X_val, y_val = extract_features_for_df(val_df, desc="Val features")

    print("\nExtracting features for Test set...")
    X_test, y_test = extract_features_for_df(test_df, desc="Test features")

    print(f"\nFeature dimensionality: {X_train.shape[1]} features per clip.")

    # Fit StandardScaler strictly on Train data to prevent data leakage
    print("Normalizing features using StandardScaler fit strictly on Train split...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    # Save feature matrices and labels
    np.savez_compressed(
        PROCESSED_DIR / "ravdess_features.npz",
        X_train=X_train_norm,
        y_train=y_train,
        X_val=X_val_norm,
        y_val=y_val,
        X_test=X_test_norm,
        y_test=y_test,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_
    )
    print(f"Features saved to {PROCESSED_DIR / 'ravdess_features.npz'}")

    # Copy demo test samples for Web UI
    copy_web_test_samples(test_df, SAMPLES_DIR)
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
