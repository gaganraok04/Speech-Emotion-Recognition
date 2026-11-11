# Speech Emotion Recognition: Model Performance & Error Analysis

## 1. Speaker-Independent Evaluation Summary
- **Overall Accuracy**: `38.33%`
- **Macro Average F1-Score**: `37.84%`
- **Test Set Composition**: 240 speech clips strictly from **Actors 21, 22, 23, and 24** (2 Male, 2 Female) who were completely unseen during training.

> **Why This Result is Honest**: In Speech Emotion Recognition (SER), models evaluated on randomly split samples often report 85–95% accuracy because they overfit to the unique vocal pitch, timbre, and acoustic footprint of individual speakers. Under a strict **Speaker-Independent Split**, benchmarks typically range between 55% and 75% on 8-class RAVDESS, representing true acoustic emotion generalization.

---

## 2. Per-Class Performance Table

| Emotion | Precision (%) | Recall (%) | F1-Score (%) | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral** | 33.33% | 37.50% | 35.29% | 16 |
| **Calm** | 31.11% | 43.75% | 36.36% | 32 |
| **Happy** | 30.95% | 40.62% | 35.14% | 32 |
| **Sad** | 36.36% | 12.50% | 18.60% | 32 |
| **Angry** | 29.63% | 50.00% | 37.21% | 32 |
| **Fearful** | 52.38% | 34.38% | 41.51% | 32 |
| **Disgust** | 47.83% | 34.38% | 40.00% | 32 |
| **Surprised** | 65.38% | 53.12% | 58.62% | 32 |

---

## 3. Top Confusion Pairs

The most frequent misclassifications observed on unseen speakers:

- **True: Sad $\rightarrow$ Predicted: Calm**: `37.5%` of instances (12 clips)
- **True: Disgust $\rightarrow$ Predicted: Angry**: `34.4%` of instances (11 clips)
- **True: Surprised $\rightarrow$ Predicted: Angry**: `34.4%` of instances (11 clips)
- **True: Neutral $\rightarrow$ Predicted: Calm**: `31.2%` of instances (5 clips)
- **True: Sad $\rightarrow$ Predicted: Happy**: `25.0%` of instances (8 clips)
- **True: Calm $\rightarrow$ Predicted: Neutral**: `18.8%` of instances (6 clips)

---

## 4. Acoustic Analysis: What Does the Model Confuse and Why?

Acoustic emotion perception is fundamentally organized around two primary psychological dimensions:
1. **Arousal** (Energy/Activation: High vs Low energy)
2. **Valence** (Positivity: Positive vs Negative)

### A. The Valence-Arousal Confusion Pattern
- **High Arousal Confusion (Angry vs Happy / Surprised)**:
  Both **Angry** and **Happy** share nearly identical spectral energy distributions: elevated pitch (F0 mean), rapid speech rate, higher RMS volume, and substantial high-frequency energy in upper MFCC bands. Because audio-only features lack semantic linguistic context, distinguishing positive high arousal (excitement/joy) from negative high arousal (anger/rage) is a classic speech challenge.
  
- **Low Arousal Confusion (Sad vs Neutral vs Calm)**:
  **Sad**, **Neutral**, and **Calm** exhibit subdued dynamics: flat pitch contours, low RMS energy, slower syllable rates, and compressed spectral contrast. While human listeners often use facial expressions to differentiate subtle sadness from neutral speech, an audio classifier relies purely on prosodic flatness, naturally leading to mutual confusion.

- **Fear vs Surprised**:
  Both exhibit sudden pitch inflections, elevated jitter, and breathiness. Without visual cues or linguistic text, acoustic cues between acute fear and sudden surprise closely overlap.

---

## 5. Mitigation & Future Work
1. **Multimodal Fusion**: Integrate acoustic features with speech-to-text (Whisper) embeddings to provide semantic valence clues.
2. **Data Augmentation**: SpecAugment (time and frequency masking), pitch shifting, and subtle background noise addition to improve generalization across vocal tracts.
3. **Self-Supervised Pretraining**: Leveraging wav2vec 2.0 or HuBERT representations pretrained on thousands of hours of unlabelled speech.
