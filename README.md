# Emotion Recognition from Speech (SER)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Feature%20Extraction-brightgreen.svg)](https://librosa.org/)
[![Web Demo](https://img.shields.io/badge/Demo-Vanilla%20HTML5%20%2F%20WebAudio-success.svg)](#-interactive-web-application--player-toolbar)

An end-to-end Speech Emotion Recognition (SER) deep learning system built with **PyTorch** and **Librosa**, featuring **strict speaker-independent train/validation/test partitioning**, comprehensive acoustic feature extraction (MFCCs, Deltas, Delta-Deltas, Chroma, Spectral Contrast, RMS), per-class performance evaluation, and an interactive **HTML5/CSS3/JavaScript Web UI** deployable to **GitHub Pages**.

---

## 🎯 The Core Assessment Requirement: Speaker-Independent Split

> **"Split by speaker, not at random — otherwise the model learns voices instead of emotions and your score is fiction."**

### Why Random Splitting Causes Fictitious Results
In Speech Emotion Recognition, randomly splitting audio clips creates severe **data leakage**:
- Every human speaker has a unique vocal anatomy: vocal tract length, fundamental pitch range ($F_0$), formant frequencies, and idiosyncratic timbral qualities.
- If clips from the same actor appear in both train and test splits, the convolutional network does not learn emotional prosody. Instead, it acts as a **speaker identifier** (e.g., memorizing that *"Speaker 14's deep voice usually says happy lines in this pitch"*).
- Random split models easily report 85%–95% test accuracy in papers and assignments, but **completely collapse on real unseen voices**.

### Our Rigorous Partitioning Strategy
We partition the 24 professional actors of the **RAVDESS** dataset strictly by Actor ID:

| Split | Actor IDs | Count | Gender Ratio | Percentage | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | Actors 01 to 16 | 960 clips | 8 Male / 8 Female | 66.7% | Feature learning & gradient updates |
| **Validation** | Actors 17 to 20 | 240 clips | 2 Male / 2 Female | 16.7% | Early stopping & hyperparameter tuning |
| **Test** | **Actors 21 to 24** | **240 clips** | **2 Male / 2 Female** | **16.7%** | **Strictly unseen benchmark voices** |

- **Zero Speaker Overlap**: Actors 21, 22, 23, and 24 are **never heard** during training or validation.
- **Strict Gender Balance**: Exactly 50% Male and 50% Female in every split to prevent gender bias.

---

## 🎛️ Acoustic Feature Extraction Explained

Speech emotion is conveyed primarily through **prosody** (pitch variations, cadence, energy, vocal tract resonance). We extract a comprehensive **280-dimensional acoustic feature vector** from each normalized audio clip:

| Feature Component | Dimension | Extraction Method | Acoustic Meaning |
| :--- | :---: | :--- | :--- |
| **MFCC Mean & Std** | 80 (40+40) | 40-band Mel filterbank across FFT frames | Spectral envelope & vocal tract shape (timbre). |
| **Delta MFCC Mean & Std** | 80 (40+40) | 1st-order temporal derivative $\frac{\partial \text{MFCC}}{\partial t}$ | Speech velocity and pitch movement rate. |
| **Delta-Delta MFCC Mean & Std** | 80 (40+40) | 2nd-order temporal derivative $\frac{\partial^2 \text{MFCC}}{\partial t^2}$ | Speech acceleration and prosodic transitions. |
| **Chroma STFT Mean & Std** | 24 (12+12) | 12 pitch classes projected from STFT | Harmonic content, intonation, and pitch chroma. |
| **Spectral Contrast Mean & Std** | 14 (7+7) | Energy diff between spectral peaks & valleys | Distinguishes clear speech from breathiness. |
| **RMS Energy Mean & Std** | 2 (1+1) | Root-Mean-Square energy across frames | Vocal loudness and volume envelope dynamics. |
| **Total** | **280** | Concatenated & Standard-Normalized | Full prosodic fingerprint of emotional speech. |

> **Data Leakage Prevention in Preprocessing**: The `StandardScaler` is fitted **strictly on the Train split**, then used to transform the Validation and Test splits.

---

## 🧠 Model Architecture

The core model is a deep **1D Convolutional Neural Network (CNN)** (`src/model.py`):
- **Conv1D Block 1**: 64 filters, kernel size 5, stride 1, BatchNorm, LeakyReLU(0.1), MaxPool1d(2), Dropout(0.25)
- **Conv1D Block 2**: 128 filters, kernel size 5, stride 1, BatchNorm, LeakyReLU(0.1), MaxPool1d(2), Dropout(0.30)
- **Conv1D Block 3**: 256 filters, kernel size 3, stride 1, BatchNorm, LeakyReLU(0.1), MaxPool1d(2), Dropout(0.35)
- **Conv1D Block 4**: 256 filters, kernel size 3, stride 1, BatchNorm, LeakyReLU(0.1), Dropout(0.35)
- **Global Average Pooling**: `AdaptiveAvgPool1d(1)` collapsing temporal dimension into invariant representations
- **Dense Classifier**: Linear(256 $\rightarrow$ 128), BatchNorm1d, ReLU, Dropout(0.40), Linear(128 $\rightarrow$ 8 classes)
- **Class-Weighted CrossEntropyLoss**: Balanced weighting to compensate for Neutral having 96 clips vs 192 clips for other emotions.

---

## 📊 Per-Class Performance & Honest Error Analysis

### 1. Per-Class Results Table (Unseen Test Actors 21–24)

| Emotion | Precision (%) | Recall (%) | F1-Score (%) | Support | Acoustic Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Surprised** | **65.4%** | **53.1%** | **58.6%** | 32 | Sudden pitch rise, high fundamental frequency ($F_0$) |
| **Fearful** | **52.4%** | **34.4%** | **41.5%** | 32 | High acoustic jitter, breathiness, vocal instability |
| **Disgust** | **47.8%** | **34.4%** | **40.0%** | 32 | Low pitch, guttural resonances, harsh timbre |
| **Angry** | **29.6%** | **50.0%** | **37.2%** | 32 | High energy, sharp onset, broad spectral dispersion |
| **Calm** | **31.1%** | **43.8%** | **36.4%** | 32 | Soft volume, slow rate, relaxed vocal folds |
| **Neutral** | **33.3%** | **37.5%** | **35.3%** | 16 | Moderate cadence, baseline spectral shape |
| **Happy** | **31.0%** | **40.6%** | **35.1%** | 32 | Bright harmonics, fast tempo, elevated pitch |
| **Sad** | **36.4%** | **12.5%** | **18.6%** | 32 | Monotone pitch, low energy, long pauses (confused with Calm/Neutral) |

- **Overall Accuracy**: `38.33%` across 8 emotions on 100% unseen test actors (vs. 12.5% random baseline, representing true cross-speaker acoustic generalization).
- **Macro Average F1**: `37.84%`

---

### 2. Honest Discussion: What Does the Model Confuse and Why?

Acoustic emotion space is organized along two psychological dimensions: **Arousal** (activation/energy) and **Valence** (positivity/negativity).

```
                 HIGH AROUSAL (Active)
                         ▲
                         │
      Angry              │             Happy
    (Negative)           │           (Positive)
                         │
  ◄──────────────────────┼──────────────────────►
  NEGATIVE VALENCE       │       POSITIVE VALENCE
                         │
      Sad                │             Calm
    (Negative)           │           (Positive)
                         │
                         ▼
                 LOW AROUSAL (Subdued)
```

1. **High Arousal Confusion (Angry $\leftrightarrow$ Happy)**:
   - **Angry** and **Happy** share virtually identical acoustic activation: elevated mean pitch ($F_0$), rapid speech rate, higher RMS volume, and abundant high-frequency energy.
   - Without linguistic text (transcript) to indicate whether the excitement is joyful or aggressive, audio-only classifiers naturally confuse positive and negative high arousal.
2. **Low Arousal Confusion (Sad $\leftrightarrow$ Neutral $\leftrightarrow$ Calm)**:
   - **Sadness**, **Neutral**, and **Calm** share flat pitch contours, low energy, and slow tempo.
   - Subtle sadness relies heavily on facial expressions; on acoustic signals alone, sadness is easily misclassified as everyday neutral speech.
3. **Fear $\leftrightarrow$ Surprise**:
   - Both exhibit abrupt pitch spikes, elevated voice jitter, and vocal breathiness.

---

## 🌐 Interactive Web UI & GitHub Pages Deployment

The `web/` directory contains a standalone client-side web application built with **HTML5, CSS3, and JavaScript** using the **Web Audio API**:
- **☕ Warm Brown & Roasted Espresso Palette**: Luxury editorial theme with instant **Dark Brown $\leftrightarrow$ Light Cream** mode switching and persistent `localStorage` memory.
- **📱 Responsive Across All Devices**: Optimized layouts for mobile phones (touch targets $\ge 44\text{px}$, kinetic table scrolling), tablets, laptops, and ultra-wide desktops.
- **🔴 Animated 3-Second Microphone Recorder**: Glowing pulse rings, live countdown timer (`3.0s` $\rightarrow$ `0.0s`), and real-time mic waveform visualization.
- **🎛️ Full Player Controls**: Dedicated **Play (▶)**, **Pause (⏸)**, **Stop (⏹)**, and **Delete (🗑️)** buttons.
- **📜 Recordings History Deck**: Automatically tracks and saves each voice clip with timestamp, detected emotion badge, and one-click replay/load.
- **Unseen Actor Test Samples**: Listen to Male and Female audio clips of unseen test speakers (Actors 21–24) and see predictions in real time.
- **Interactive 8×8 Confusion Matrix**: Hover and click on any cell to see acoustic confusion explanations.

### Deploying to GitHub Pages in 3 Steps:
1. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Speech Emotion Recognition project"
   git remote add origin https://github.com/<your-username>/emotion-recognition-speech.git
   git branch -M main
   git push -u origin main
   ```
2. In your GitHub repository, open **Settings** $\rightarrow$ **Pages**.
3. Under **Build and deployment** $\rightarrow$ **Branch**, select `main` and folder `/web`, then click **Save**.
4. Your interactive web demo will be live at `https://<your-username>.github.io/emotion-recognition-speech/`!

---

## 🚀 Quickstart: Reproduce Everything from Scratch

### 1. Local Environment & Dependencies
```bash
cd "emotion recognition speech"
pip install -r requirements.txt
```

### 2. Download RAVDESS Dataset (~214 MB)
```bash
python scripts/download_ravdess.py
```

### 3. Extract Features & Split by Speaker
```bash
python -m src.preprocess
```

### 4. Train Deep CNN Model
```bash
python -m src.train --model cnn1d --epochs 60 --batch-size 32
```

### 5. Generate Evaluation Report & Figures
```bash
python -m src.evaluate
```

### 6. Generate 1-Minute LinkedIn Showcase Video (1080p with Female Voice & Burned Subtitles)
```bash
python scripts/generate_video.py
```
Outputs high-definition video [`video/emotion_recognition_linkedin_1min.mp4`](video/emotion_recognition_linkedin_1min.mp4) along with ready-to-post LinkedIn text in [`video/LINKEDIN_POST.md`](video/LINKEDIN_POST.md).

---

## 📁 Repository Structure

```text
emotion-recognition-speech/
├── data/
│   ├── raw/                 # Extracted RAVDESS audio (Actors 01 to 24)
│   └── processed/           # Cached features (.npz) and metadata splits
├── models/
│   ├── best_cnn1d.pt        # Saved PyTorch checkpoint with highest Val F1
│   └── emotion_cnn1d.onnx   # Production ONNX model export
├── notebooks/
│   ├── 01_EDA_and_Audio_Visualization.ipynb  # Interactive EDA & feature spectrograms
│   └── visualize_audio_features.py           # Feature analysis & plot generator
├── reports/
│   ├── per_class_metrics.csv # Precision, Recall, F1, Support table
│   ├── per_class_metrics.json
│   ├── confusion_matrix.png # High-resolution confusion matrix heatmap
│   └── ERROR_ANALYSIS.md    # Detailed acoustic discussion of failure modes
├── scripts/
│   ├── download_ravdess.py  # Automated RAVDESS downloader & validator
│   ├── generate_video.py    # 1080p video synthesizer with Edge-TTS voiceover & subtitles
│   └── serve_web.py         # Local development server for web UI
├── src/
│   ├── dataset.py           # Speaker-independent partitioning (Actors 21-24 test)
│   ├── features.py          # 280-D MFCC, Delta, Chroma, Contrast, RMS extraction
│   ├── model.py             # 1D CNN & CRNN architectures in PyTorch
│   ├── preprocess.py        # Feature extraction pipeline & test sample bundler
│   ├── train.py             # Training loop with class-weighted CrossEntropy
│   └── evaluate.py          # Classification report & confusion matrix generator
├── tests/
│   └── test_pipeline.py     # 4/4 passing unit tests verifying zero data leakage
├── video/
│   ├── emotion_recognition_linkedin_1min.mp4  # 1-min 1080p LinkedIn showcase video
│   ├── LINKEDIN_POST.md                       # Ready-to-copy LinkedIn caption
│   └── subtitles.srt                          # Synchronized subtitle captions
├── web/                     # GitHub Pages static site (Espresso & Cashmere Theme)
│   ├── index.html           # Modern responsive layout (mobile, tablet, laptop)
│   ├── style.css            # Light & Dark warm brown theme
│   ├── app.js               # Audio playback, 3s recording, waveform visualizer, history
│   ├── emotion_cnn1d.onnx   # Browser-ready ONNX model
│   ├── metrics.json         # Live per-class scores
│   ├── confusion_matrix.json
│   └── samples/             # Bundled audio test clips from unseen test actors
├── requirements.txt         # Pinned Python dependencies
└── README.md                # Project documentation
```
