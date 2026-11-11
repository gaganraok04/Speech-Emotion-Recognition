"""
Download and extract the RAVDESS Audio Speech Dataset.
Source: HuggingFace mirror of Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
Dataset contains audio speech across 24 actors (12 male, 12 female) uttering statements with 8 emotions.
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

HF_ZIP_URL = "https://huggingface.co/datasets/viks66/ravdess_speech/resolve/main/ravdess.zip"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = DATA_DIR / "ravdess_speech.zip"


def download_file(url: str, dest_path: Path):
    print(f"Connecting to: {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 128  # 128 KB chunks

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print("\nDownload successfully completed!")


def extract_zip(zip_path: Path, target_dir: Path):
    print(f"Extracting {zip_path.name} to {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    print("Extraction successfully completed!")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_wavs = list(RAW_DIR.glob("**/*.wav"))
    if len(existing_wavs) >= 1400:
        print(f"Found {len(existing_wavs)} audio files in {RAW_DIR}. Dataset is already downloaded.")
        return

    # Check if zip exists
    if not ZIP_PATH.exists() or ZIP_PATH.stat().st_size < 200_000_000:
        print("Dataset zip not found or incomplete. Downloading RAVDESS (~214 MB)...")
        download_file(HF_ZIP_URL, ZIP_PATH)
    else:
        print(f"Found existing valid archive at {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024*1024):.1f} MB).")

    extract_zip(ZIP_PATH, RAW_DIR)

    wav_count = len(list(RAW_DIR.glob("**/*.wav")))
    print(f"Total audio clips verified: {wav_count}")


if __name__ == "__main__":
    main()
