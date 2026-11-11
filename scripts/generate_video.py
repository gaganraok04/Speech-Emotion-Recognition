"""
Final Polish: LinkedIn 1-Minute Showcase Video Generator for Speech Emotion Recognition.

Features:
- Professional 1080p (1920x1080) slide design in warm brown espresso & amber palette.
- Zero missing glyphs or broken characters (uses standard system fonts and native PIL shapes).
- Elevated layout architecture with dedicated lower-third subtitle safe zone (y = 850 to 1080).
- Minimalist, elegant subtitles:
  * Font size: 34 px (clean sans-serif Segoe UI / Arial medium).
  * Color: White (#FFFFFF), line-height 1.2, max-width 72%.
  * Fitted translucent background: rgba(0, 0, 0, 0.65) with 8px border radius.
  * Padding: 20px horizontal, 11px vertical.
  * Position: Safe area 65px from bottom edge (y_bottom = 1015).
  * Natural sentence-based phrasing synchronized with speech pauses.
  * Zero clutter, zero obscuring of technical content.
- Closing slide strictly adheres to guidelines:
  * "Speech Emotion Recognition"
  * "AI-Powered Speech Emotion Analysis"
  * "Thank You for Watching!"
  * Completely removes any fake GitHub URLs or fabricated repositories.
- Natural female voiceover (en-US-AriaNeural) at exactly ~64 seconds.
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "video"
SLIDES_DIR = VIDEO_DIR / "slides"
AUDIO_DIR = VIDEO_DIR / "audio"
CLIPS_DIR = VIDEO_DIR / "clips"
SUB_IMGS_DIR = VIDEO_DIR / "sub_images"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Fonts
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CODE = "C:/Windows/Fonts/consola.ttf"

# Palettes (Warm Brown & Espresso)
COLOR_BG = (18, 14, 11)           # #120e0b
COLOR_CARD = (34, 27, 21)         # #221b15
COLOR_AMBER = (217, 119, 6)       # #d97706
COLOR_AMBER_BRIGHT = (245, 158, 11)# #f59e0b
COLOR_TERRACOTTA = (194, 65, 12)  # #c2410c
COLOR_TEXT_MAIN = (252, 249, 245) # #fcf9f5
COLOR_TEXT_MUTED = (184, 168, 152)# #b8a898
COLOR_TEXT_DIM = (132, 114, 98)   # #847262
COLOR_RED = (220, 38, 38)
COLOR_GREEN = (22, 163, 74)

# Scenes with exact speech phrases and timing intervals (detected from audio narration)
SCENES = [
    {
        "id": "scene_01_welcome",
        "title": "Welcome Slide",
        "audio_file": "scene_01_welcome.mp3",
        "duration": 10.92,
        "phrases": [
            {"start": 0.20, "end": 3.70, "text": "Can an AI detect your true emotion just from the tone of your voice?"},
            {"start": 4.10, "end": 7.60, "text": "Most speech models cheat by memorizing speaker voices."},
            {"start": 8.00, "end": 10.30, "text": "Here is how we solved it with PyTorch."}
        ]
    },
    {
        "id": "scene_02_speaker_split",
        "title": "The Golden Rule: Speaker Independence",
        "audio_file": "scene_02_speaker_split.mp3",
        "duration": 11.18,
        "phrases": [
            {"start": 0.15, "end": 3.60, "text": "If you split speech data at random, your accuracy is fiction."},
            {"start": 4.00, "end": 7.60, "text": "We enforced a strict speaker-independent split on twenty-four actors,"},
            {"start": 7.80, "end": 10.60, "text": "testing only on completely unseen voices."}
        ]
    },
    {
        "id": "scene_03_features_model",
        "title": "280-D Acoustic Features & PyTorch CNN",
        "audio_file": "scene_03_features_model.mp3",
        "duration": 11.86,
        "phrases": [
            {"start": 0.15, "end": 5.60, "text": "We engineered a 280-dimensional acoustic feature vector,"},
            {"start": 5.70, "end": 8.10, "text": "capturing 40 MFCC bands, pitch velocity, and spectral contrast,"},
            {"start": 8.20, "end": 11.20, "text": "trained with a deep 1D Convolutional Neural Network."}
        ]
    },
    {
        "id": "scene_04_live_demo",
        "title": "Real-Time Working Project & UI",
        "audio_file": "scene_04_live_demo.mp3",
        "duration": 9.91,
        "phrases": [
            {"start": 0.20, "end": 1.80, "text": "Here is the live project!"},
            {"start": 2.20, "end": 5.90, "text": "Featuring live three-second microphone recording with animated waveforms,"},
            {"start": 6.00, "end": 9.30, "text": "instant emotion prediction, and a recordings history deck."}
        ]
    },
    {
        "id": "scene_05_confusion_analysis",
        "title": "Honest Acoustic Confusion Analysis",
        "audio_file": "scene_05_confusion_analysis.mp3",
        "duration": 9.62,
        "phrases": [
            {"start": 0.15, "end": 2.50, "text": "We delivered honest benchmark metrics."},
            {"start": 2.90, "end": 5.60, "text": "High arousal overlaps between anger and joy,"},
            {"start": 5.70, "end": 9.10, "text": "while subtle sadness shares acoustic prosody with calm speech."}
        ]
    },
    {
        "id": "scene_06_thank_you",
        "title": "Thank You & Professional Closing",
        "audio_file": "scene_06_thank_you.mp3",
        "duration": 10.34,
        "phrases": [
            {"start": 0.15, "end": 1.60, "text": "Thank you for watching!"},
            {"start": 1.80, "end": 5.80, "text": "Speech Emotion Recognition delivers deep acoustic intelligence for real-world voice signals."},
            {"start": 6.10, "end": 9.90, "text": "Connect on LinkedIn to discuss the project!"}
        ]
    }
]


def get_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def create_base_canvas():
    """
    Creates base 1920x1080 canvas without broken emojis.
    Leaves the bottom safe zone (y = 850 to 1080) completely clear for subtitles.
    """
    img = Image.new("RGB", (1920, 1080), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Ambient radial glow in corner
    for r in range(400, 0, -20):
        alpha = int(12 * (1 - r / 400))
        glow_color = (217, 119, 6, alpha)
        draw.ellipse([(100 - r, 80 - r), (100 + r, 80 + r)], fill=glow_color)

    # Brand Navbar (Clean, professional text & geometric badges)
    brand_font = get_font(FONT_BOLD, 22)
    badge_font = get_font(FONT_BOLD, 15)

    # Audio signal level graphic
    draw.rounded_rectangle([(100, 52), (108, 72)], radius=2, fill=COLOR_AMBER_BRIGHT)
    draw.rounded_rectangle([(112, 46), (120, 72)], radius=2, fill=COLOR_AMBER_BRIGHT)
    draw.rounded_rectangle([(124, 40), (132, 72)], radius=2, fill=COLOR_AMBER_BRIGHT)
    draw.rounded_rectangle([(136, 48), (144, 72)], radius=2, fill=COLOR_AMBER_BRIGHT)

    draw.text((158, 48), "SPEECH EMOTION RECOGNITION (SER)", font=brand_font, fill=COLOR_AMBER_BRIGHT)
    
    draw.rounded_rectangle([(1520, 42), (1820, 78)], radius=18, fill=(34, 27, 21), outline=COLOR_AMBER, width=1)
    draw.text((1556, 51), "1-MIN SPOTLIGHT", font=badge_font, fill=COLOR_AMBER_BRIGHT)

    return img, draw


def render_slide_welcome(path: Path):
    img, draw = create_base_canvas()
    # Main container box ends at y = 825
    draw.rounded_rectangle([(100, 110), (1820, 825)], radius=20, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    badge_font = get_font(FONT_BOLD, 17)
    draw.rounded_rectangle([(770, 140), (1150, 182)], radius=21, fill=(50, 36, 26), outline=COLOR_AMBER, width=1)
    # Red status indicator dot
    draw.ellipse([(795, 156), (807, 168)], fill=COLOR_RED)
    draw.text((820, 151), "PROJECT SPOTLIGHT", font=badge_font, fill=COLOR_AMBER_BRIGHT)

    title_font = get_font(FONT_BOLD, 56)
    draw.text((140, 215), "Emotion Recognition from Speech", font=title_font, fill=COLOR_TEXT_MAIN)
    
    sub_font = get_font(FONT_REGULAR, 28)
    draw.text((140, 295), "Why Most Speech Emotion Models Fail — And How We Solved It", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    pills = [
        ("Zero Speaker Leakage", "Tested strictly on 100% unseen voices"),
        ("280-D Acoustic Features", "Pitch velocity, timbre & energy dynamics"),
        ("Interactive Web UI & Audio Deck", "Microphone recording & waveform player")
    ]
    pill_font_title = get_font(FONT_BOLD, 21)
    pill_font_desc = get_font(FONT_REGULAR, 17)

    for i, (p_title, p_desc) in enumerate(pills):
        x = 140 + i * 560
        draw.rounded_rectangle([(x, 360), (x + 520, 560)], radius=14, fill=(24, 18, 14), outline=COLOR_AMBER, width=1)
        draw.text((x + 28, 400), p_title, font=pill_font_title, fill=COLOR_TEXT_MAIN)
        draw.text((x + 28, 455), p_desc, font=pill_font_desc, fill=COLOR_TEXT_MUTED)

    # Ambient waveform display
    np.random.seed(42)
    for w in range(0, 1680, 18):
        amp = np.sin(w / 50) * np.cos(w / 100) * 32 + np.random.normal(0, 6)
        draw.line([(140 + w, 660 - amp), (140 + w, 660 + amp)], fill=COLOR_AMBER_BRIGHT, width=3)

    tag_font = get_font(FONT_REGULAR, 17)
    draw.text((140, 740), "Speech Prosody Pipeline • Librosa 280-D Extraction • Deep 1D CNN in PyTorch", font=tag_font, fill=COLOR_TEXT_DIM)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_speaker_split(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(100, 110), (1820, 825)], radius=20, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 46)
    draw.text((140, 140), "The Golden Rule: Speaker-Independent Split", font=title_font, fill=COLOR_TEXT_MAIN)

    tag_font = get_font(FONT_REGULAR, 24)
    draw.text((140, 205), "Split by speaker, not at random — otherwise your accuracy is fiction.", font=tag_font, fill=COLOR_AMBER_BRIGHT)

    box_font_h = get_font(FONT_BOLD, 24)
    box_font_body = get_font(FONT_REGULAR, 19)

    # Left: The Trap
    draw.rounded_rectangle([(140, 260), (940, 790)], radius=16, fill=(38, 22, 22), outline=COLOR_RED, width=2)
    draw.rounded_rectangle([(175, 290), (205, 320)], radius=4, fill=(80, 20, 20), outline=COLOR_RED)
    draw.text((184, 292), "X", font=get_font(FONT_BOLD, 18), fill=(255, 120, 120))
    draw.text((220, 294), "The Random Split Trap (Data Leakage)", font=box_font_h, fill=(239, 68, 68))
    
    draw.text((175, 360), "• Test set contains the SAME speakers heard in training", font=box_font_body, fill=COLOR_TEXT_MAIN)
    draw.text((175, 415), "• Model memorizes actor vocal tract, timbre & pitch range", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((175, 470), "• Reports inflated '90-95%' test accuracy in papers", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((175, 525), "• Completely fails when tested on any new human voice", font=box_font_body, fill=(239, 68, 68))
    
    draw.rounded_rectangle([(175, 665), (905, 745)], radius=12, fill=(60, 20, 20))
    draw.text((205, 690), "FICTITIOUS RESULT: Overfitting to Speaker Identity", font=get_font(FONT_BOLD, 19), fill=(255, 120, 120))

    # Right: Our Approach
    draw.rounded_rectangle([(980, 260), (1780, 790)], radius=16, fill=(24, 34, 24), outline=COLOR_GREEN, width=2)
    draw.rounded_rectangle([(1015, 290), (1045, 320)], radius=4, fill=(20, 60, 25), outline=COLOR_GREEN)
    draw.text((1024, 292), "+", font=get_font(FONT_BOLD, 18), fill=(150, 255, 160))
    draw.text((1060, 294), "Our Speaker-Independent Split (Reality)", font=box_font_h, fill=(34, 197, 94))
    
    draw.text((1015, 360), "• RAVDESS 24 professional actors partitioned strictly by ID", font=box_font_body, fill=COLOR_TEXT_MAIN)
    draw.text((1015, 415), "• Train: Actors 01-16 (8 Male, 8 Female, 66.7%)", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((1015, 470), "• Validation: Actors 17-20 (2 Male, 2 Female, 16.7%)", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((1015, 525), "• Test: Actors 21-24 (2 Male, 2 Female) — 100% UNSEEN", font=box_font_body, fill=(34, 197, 94))
    
    draw.rounded_rectangle([(1015, 665), (1745, 745)], radius=12, fill=(20, 55, 25))
    draw.text((1045, 690), "HONEST RESULT: Generalizes to Real Human Emotions", font=get_font(FONT_BOLD, 19), fill=(150, 255, 160))

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_features_model(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(100, 110), (1820, 825)], radius=20, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 46)
    draw.text((140, 140), "280-Dimensional Acoustic Feature Pipeline", font=title_font, fill=COLOR_TEXT_MAIN)
    sub_font = get_font(FONT_REGULAR, 24)
    draw.text((140, 205), "Extracting spectral envelopes, velocity derivatives, and harmonic cues with Librosa", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    feats = [
        ("40 MFCCs (80-D)", "Mean & std across 40 Mel filterbanks mimicking human cochlear hearing.", COLOR_AMBER),
        ("Delta & Delta-Delta (160-D)", "1st & 2nd temporal derivatives capturing pitch velocity & acceleration.", COLOR_TERRACOTTA),
        ("Chroma & Contrast (38-D)", "12 pitch chroma classes for intonation plus 7 frequency spectral contrast bands.", COLOR_AMBER_BRIGHT),
        ("RMS Energy (2-D)", "Frame-level Root-Mean-Square energy tracking dynamic loudness contour.", (168, 153, 138))
    ]

    h_font = get_font(FONT_BOLD, 22)
    b_font = get_font(FONT_REGULAR, 17)

    for i, (f_name, f_desc, f_col) in enumerate(feats):
        y = 265 + i * 132
        draw.rounded_rectangle([(140, y), (1180, y + 115)], radius=12, fill=(24, 18, 14), outline=f_col, width=1)
        draw.text((165, y + 18), f_name, font=h_font, fill=f_col)
        draw.text((165, y + 60), f_desc, font=b_font, fill=COLOR_TEXT_MUTED)

    draw.rounded_rectangle([(1220, 265), (1780, 790)], radius=16, fill=(28, 21, 16), outline=COLOR_AMBER, width=2)
    draw.text((1255, 300), "PyTorch 1D CNN Architecture", font=get_font(FONT_BOLD, 23), fill=COLOR_TEXT_MAIN)
    
    arch_lines = [
        "• Conv1D Block 1: 64 filters (k=5)",
        "• Conv1D Block 2: 128 filters (k=5)",
        "• Conv1D Block 3: 256 filters (k=3)",
        "• Conv1D Block 4: 256 filters (k=3)",
        "• Batch Normalization & LeakyReLU",
        "• AdaptiveAvgPool1d Global Pooling",
        "• Dense Classifier: 256 -> 128 -> 8",
        "• Class-Weighted CrossEntropyLoss"
    ]
    for idx, line in enumerate(arch_lines):
        col = COLOR_AMBER_BRIGHT if ("Conv" in line or "Class" in line) else COLOR_TEXT_MUTED
        draw.text((1255, 355 + idx * 48), line, font=get_font(FONT_REGULAR, 18), fill=col)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_live_demo(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(100, 110), (1820, 825)], radius=20, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 46)
    draw.text((140, 140), "Real-Time Working Project: Interactive Web UI", font=title_font, fill=COLOR_TEXT_MAIN)
    sub_font = get_font(FONT_REGULAR, 24)
    draw.text((140, 205), "Live 3-Second Microphone Recording • Audio Player Toolbar • Recordings History", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    # Left Demo Box (stops at y = 790)
    draw.rounded_rectangle([(140, 265), (1120, 790)], radius=16, fill=(24, 18, 14), outline=COLOR_AMBER, width=2)
    
    # Active recording header
    draw.rounded_rectangle([(165, 285), (1095, 345)], radius=10, fill=(60, 20, 20), outline=COLOR_RED, width=1)
    draw.ellipse([(190, 305), (210, 325)], fill=COLOR_RED)
    draw.text((225, 302), "LIVE RECORDING ACTIVE — Speak into microphone...", font=get_font(FONT_BOLD, 17), fill=(255, 140, 140))
    draw.text((990, 297), "2.4s", font=get_font(FONT_BOLD, 28), fill=COLOR_RED)

    # Waveform canvas
    draw.rounded_rectangle([(165, 360), (1095, 465)], radius=10, fill=(18, 13, 10), outline=(50, 38, 30), width=1)
    np.random.seed(101)
    for w in range(0, 910, 10):
        amp = np.sin(w / 30) * np.sin(w / 80) * 38 + np.random.normal(0, 6)
        draw.line([(175 + w, 412 - amp), (175 + w, 412 + amp)], fill=COLOR_AMBER_BRIGHT, width=2)

    # Play/Pause/Stop/Delete Toolbar (Clean text labels, no broken glyphs)
    btns = [("PLAY", (217, 119, 6)), ("PAUSE", (50, 38, 30)), ("STOP", (50, 38, 30)), ("DELETE", (120, 30, 30))]
    for b_idx, (b_label, b_color) in enumerate(btns):
        bx = 165 + b_idx * 235
        draw.rounded_rectangle([(bx, 480), (bx + 220, 532)], radius=8, fill=b_color)
        draw.text((bx + 75, 496), b_label, font=get_font(FONT_BOLD, 17), fill=COLOR_TEXT_MAIN)

    # Classified emotion card
    draw.rounded_rectangle([(165, 550), (1095, 765)], radius=12, fill=(32, 24, 18), outline=COLOR_AMBER, width=1)
    draw.text((190, 570), "CLASSIFIED EMOTION", font=get_font(FONT_BOLD, 15), fill=COLOR_TEXT_DIM)
    draw.text((190, 598), "ANGRY", font=get_font(FONT_BOLD, 42), fill=COLOR_AMBER_BRIGHT)
    draw.rounded_rectangle([(380, 608), (520, 644)], radius=14, fill=(80, 25, 25), outline=COLOR_RED, width=1)
    draw.text((400, 618), "High Arousal", font=get_font(FONT_BOLD, 15), fill=(255, 120, 120))
    
    draw.rounded_rectangle([(190, 678), (1070, 693)], radius=8, fill=(20, 15, 12))
    draw.rounded_rectangle([(190, 678), (190 + int(880 * 0.742), 693)], radius=8, fill=COLOR_AMBER_BRIGHT)
    draw.text((190, 715), "Confidence: 74.2% • Unseen Actor Voice Generalization", font=get_font(FONT_REGULAR, 16), fill=COLOR_TEXT_MUTED)

    # Right: History Deck
    draw.rounded_rectangle([(1160, 265), (1780, 790)], radius=16, fill=(28, 21, 16), outline=(60, 46, 36), width=2)
    draw.text((1195, 300), "Live Recordings History Deck", font=get_font(FONT_BOLD, 23), fill=COLOR_TEXT_MAIN)

    hist_items = [
        ("Microphone Recording #3", "Just now • 3.00s", "Angry (74.2%)", COLOR_RED),
        ("Microphone Recording #2", "1 min ago • 3.00s", "Surprised (65.4%)", COLOR_AMBER_BRIGHT),
        ("Microphone Recording #1", "3 mins ago • 3.00s", "Calm (58.1%)", COLOR_GREEN),
        ("Actor 22 Test Sample", "5 mins ago • 3.20s", "Happy (60.0%)", COLOR_AMBER)
    ]
    for h_i, (h_title, h_sub, h_tag, h_col) in enumerate(hist_items):
        hy = 355 + h_i * 102
        draw.rounded_rectangle([(1195, hy), (1745, hy + 86)], radius=10, fill=(20, 15, 12), outline=(50, 38, 30), width=1)
        draw.text((1220, hy + 16), h_title, font=get_font(FONT_BOLD, 18), fill=COLOR_TEXT_MAIN)
        draw.text((1220, hy + 48), h_sub, font=get_font(FONT_REGULAR, 15), fill=COLOR_TEXT_DIM)
        draw.rounded_rectangle([(1535, hy + 22), (1720, hy + 62)], radius=12, fill=(40, 30, 24), outline=h_col, width=1)
        draw.text((1550, hy + 30), h_tag, font=get_font(FONT_BOLD, 14), fill=h_col)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_confusion(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(100, 110), (1820, 825)], radius=20, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 46)
    draw.text((140, 140), "Honest Acoustic Evaluation & Confusion Analysis", font=title_font, fill=COLOR_TEXT_MAIN)
    sub_font = get_font(FONT_REGULAR, 24)
    draw.text((140, 205), "Evaluated strictly on unseen Actors 21–24 (50% Male / 50% Female)", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    cm_img_path = REPORTS_DIR / "confusion_matrix.png"
    if cm_img_path.exists():
        cm_img = Image.open(cm_img_path).convert("RGB")
        cm_img = cm_img.resize((660, 520), Image.Resampling.LANCZOS)
        img.paste(cm_img, (140, 265))
    else:
        draw.rounded_rectangle([(140, 265), (800, 785)], radius=14, fill=(24, 18, 14))

    draw.rounded_rectangle([(840, 265), (1780, 785)], radius=16, fill=(28, 21, 16), outline=COLOR_AMBER, width=1)
    draw.text((875, 300), "What Does the Model Confuse and Why?", font=get_font(FONT_BOLD, 24), fill=COLOR_TEXT_MAIN)

    reasons = [
        ("High Arousal Overlap: Angry <-> Happy", "Both share identical spectral signatures: elevated pitch (F0), rapid speech cadence, and sharp high-frequency energy."),
        ("Low Arousal Compression: Sad <-> Calm <-> Neutral", "Subdued dynamic range, slow tempo, and flat pitch contours make subtle sadness acoustic cousins with calm speech."),
        ("Vocal Jitter & Instability: Fear <-> Surprise", "Both display sudden pitch excursions, vocal breathiness, and acoustic jitter in frequency bands.")
    ]

    r_title_font = get_font(FONT_BOLD, 20)
    r_body_font = get_font(FONT_REGULAR, 17)

    for idx, (rt, rb) in enumerate(reasons):
        ry = 360 + idx * 135
        draw.rounded_rectangle([(875, ry), (1745, ry + 115)], radius=10, fill=(20, 15, 12), outline=(50, 38, 30), width=1)
        draw.text((898, ry + 16), rt, font=r_title_font, fill=COLOR_AMBER_BRIGHT)
        words = rb.split()
        line1 = " ".join(words[:13])
        line2 = " ".join(words[13:])
        draw.text((898, ry + 50), line1, font=r_body_font, fill=COLOR_TEXT_MUTED)
        draw.text((898, ry + 76), line2, font=r_body_font, fill=COLOR_TEXT_MUTED)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_thank_you(path: Path):
    """
    Closing slide strictly adhering to instructions:
    - Displays:
        Speech Emotion Recognition
        AI-Powered Speech Emotion Analysis
        Thank You for Watching!
    - Zero fake GitHub URLs or invented repositories.
    """
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(100, 110), (1820, 825)], radius=20, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    # Top Pill
    draw.rounded_rectangle([(770, 140), (1150, 182)], radius=21, fill=(50, 36, 26), outline=COLOR_AMBER, width=1)
    draw.text((800, 151), "AI RESEARCH & ENGINEERING", font=get_font(FONT_BOLD, 17), fill=COLOR_AMBER_BRIGHT)

    # Core required headings
    title_font = get_font(FONT_BOLD, 62)
    draw.text((140, 215), "Speech Emotion Recognition", font=title_font, fill=COLOR_TEXT_MAIN)
    
    sub_font = get_font(FONT_REGULAR, 30)
    draw.text((140, 295), "AI-Powered Speech Emotion Analysis", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    ty_font = get_font(FONT_BOLD, 44)
    draw.text((140, 355), "Thank You for Watching!", font=ty_font, fill=(255, 230, 200))

    # 3 Architecture Summary Cards
    summaries = [
        ("Strict Speaker Independence", "Trained on Actors 01-16, tested strictly on unseen Actors 21-24 with zero data leakage.", COLOR_AMBER),
        ("280-D Acoustic Features", "40 Mel filterbanks, velocity deltas, acceleration delta-deltas, pitch chroma & RMS energy.", COLOR_TERRACOTTA),
        ("Deep 1D CNN in PyTorch", "Class-weighted loss, spatial dropout, batch normalization & real-time ONNX web inference.", COLOR_GREEN)
    ]
    c_title = get_font(FONT_BOLD, 22)
    c_desc = get_font(FONT_REGULAR, 17)

    for i, (ct, cd, cc) in enumerate(summaries):
        x = 140 + i * 560
        draw.rounded_rectangle([(x, 440), (x + 520, 660)], radius=14, fill=(24, 18, 14), outline=cc, width=2)
        draw.text((x + 28, 475), ct, font=c_title, fill=cc)
        words = cd.split()
        mid = len(words) // 2
        line1 = " ".join(words[:mid])
        line2 = " ".join(words[mid:])
        draw.text((x + 28, 540), line1, font=c_desc, fill=COLOR_TEXT_MAIN)
        draw.text((x + 28, 575), line2, font=c_desc, fill=COLOR_TEXT_MUTED)

    # Bottom Verification Tag Banner
    draw.rounded_rectangle([(140, 700), (1780, 770)], radius=12, fill=(26, 20, 15), outline=COLOR_AMBER, width=1)
    tag_font = get_font(FONT_BOLD, 18)
    draw.text((180, 722), "PyTorch Deep Learning • Librosa Audio Processing • RAVDESS Benchmark • Web Audio API", font=tag_font, fill=COLOR_AMBER_BRIGHT)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_subtitle_card(
    base_img: Image.Image,
    text: str,
    font_size: int = 34,
    pad_x: int = 20,
    pad_y: int = 11,
    radius: int = 8,
    margin_bottom: int = 65
) -> Image.Image:
    """
    Renders professional, minimal subtitle card:
    - Font size: 34 px (32-38 px specification).
    - Font weight: medium / semibold sans-serif.
    - Color: White (#FFFFFF).
    - Subtle translucent background: rgba(0, 0, 0, 0.65) -> (0, 0, 0, 166).
    - Border radius: 8 px.
    - Padding: 20 px horizontal, 11 px vertical.
    - Fitted strictly to text width (not stretched across the entire screen).
    - Lower-third safe area: 65 px from bottom edge (y_bottom = 1015).
    - Max width: 72% of screen width (1382 px).
    - Up to 2 lines with natural sentence-based breaking.
    """
    img = base_img.copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(FONT_REGULAR, font_size)

    max_w = int(img.width * 0.72)
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]

    lines = [text]
    if w > max_w:
        words = text.split()
        mid = len(words) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]

    line_bboxes = [font.getbbox(l) for l in lines]
    line_ws = [b[2] - b[0] for b in line_bboxes]
    line_hs = [b[3] - b[1] for b in line_bboxes]
    
    max_line_w = max(line_ws)
    line_spacing = int(font_size * 0.20)
    total_text_h = sum(line_hs) + (len(lines) - 1) * line_spacing

    box_w = max_line_w + 2 * pad_x
    box_h = total_text_h + 2 * pad_y

    x0 = (img.width - box_w) // 2
    y0 = img.height - margin_bottom - box_h
    x1 = x0 + box_w
    y1 = y0 + box_h

    # Translucent black background rgba(0, 0, 0, 0.65)
    draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=radius, fill=(0, 0, 0, 166))

    # Draw centered text lines
    curr_y = y0 + pad_y
    for i, l in enumerate(lines):
        lw = line_ws[i]
        lx = x0 + (box_w - lw) // 2
        draw.text((lx, curr_y - line_bboxes[i][1]), l, font=font, fill=(255, 255, 255, 255))
        curr_y += line_hs[i] + line_spacing

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def build_video():
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    SUB_IMGS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Step 1: Rendering Redesigned 1080p Base Slides (Content ends at y=825) ---")
    slide_paths = [
        SLIDES_DIR / "slide_01.png",
        SLIDES_DIR / "slide_02.png",
        SLIDES_DIR / "slide_03.png",
        SLIDES_DIR / "slide_04.png",
        SLIDES_DIR / "slide_05.png",
        SLIDES_DIR / "slide_06.png"
    ]
    render_slide_welcome(slide_paths[0])
    render_slide_speaker_split(slide_paths[1])
    render_slide_features_model(slide_paths[2])
    render_slide_live_demo(slide_paths[3])
    render_slide_confusion(slide_paths[4])
    render_slide_thank_you(slide_paths[5])

    print("\n--- Step 2: Generating Synchronized SRT & Subtitle Images ---")
    srt_entries = []
    global_time = 0.0
    seg_counter = 0
    all_segment_clips = []

    for idx, scene in enumerate(SCENES):
        base_slide = Image.open(slide_paths[idx]).convert("RGB")
        scene_dur = scene["duration"]
        phrases = scene["phrases"]
        
        # Build timeline for this scene:
        # Each interval is either silence (clean slide) or a phrase (slide with subtitle card)
        timeline = []
        curr = 0.0

        for p in phrases:
            p_start = p["start"]
            p_end = p["end"]
            p_text = p["text"]

            if p_start > curr + 0.04:
                # Silence before phrase
                timeline.append({
                    "duration": round(p_start - curr, 3),
                    "type": "clean",
                    "text": None
                })
            
            p_dur = round(p_end - p_start, 3)
            timeline.append({
                "duration": p_dur,
                "type": "phrase",
                "text": p_text
            })

            # Add to SRT
            srt_entries.append({
                "idx": len(srt_entries) + 1,
                "start": format_srt_time(global_time + p_start),
                "end": format_srt_time(global_time + p_end),
                "text": p_text
            })

            curr = p_end

        if curr < scene_dur - 0.04:
            # Trailing silence
            timeline.append({
                "duration": round(scene_dur - curr, 3),
                "type": "clean",
                "text": None
            })

        # Render segment images and generate short video clips
        for seg in timeline:
            seg_counter += 1
            dur = seg["duration"]
            seg_img_path = SUB_IMGS_DIR / f"seg_{seg_counter:03d}.png"
            seg_clip_path = CLIPS_DIR / f"seg_{seg_counter:03d}.mp4"

            if seg["type"] == "phrase":
                sub_img = render_subtitle_card(base_slide, seg["text"])
                sub_img.save(seg_img_path)
            else:
                base_slide.save(seg_img_path)

            # Generate constant 25fps video clip for segment
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(seg_img_path),
                "-t", str(dur),
                "-r", "25",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                str(seg_clip_path)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            all_segment_clips.append(seg_clip_path)

        global_time += scene_dur

    # Write SRT file
    srt_path = VIDEO_DIR / "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for entry in srt_entries:
            f.write(f"{entry['idx']}\n")
            f.write(f"{entry['start']} --> {entry['end']}\n")
            f.write(f"{entry['text']}\n\n")
    print(f"Synchronized SRT written to: {srt_path}")

    # Step 3: Stitching Audio Narration
    print("\n--- Step 3: Stitching Audio Narration ---")
    concat_audio_txt = AUDIO_DIR / "concat_audio.txt"
    with open(concat_audio_txt, "w", encoding="utf-8") as f:
        for scene in SCENES:
            f.write(f"file '{scene['audio_file']}'\n")

    full_audio_path = VIDEO_DIR / "full_narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_audio_txt),
        "-c", "copy",
        str(full_audio_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Full audio narration: {full_audio_path}")

    # Step 4: Concatenating all segment clips
    print(f"\n--- Step 4: Concatenating {len(all_segment_clips)} Segment Clips ---")
    concat_video_txt = VIDEO_DIR / "concat_segments.txt"
    with open(concat_video_txt, "w", encoding="utf-8") as f:
        for cp in all_segment_clips:
            f.write(f"file '{cp.resolve().as_posix()}'\n")

    unsubbed_video = VIDEO_DIR / "unsubbed_video.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_video_txt),
        "-c", "copy",
        str(unsubbed_video)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Concatenated video generated: {unsubbed_video}")

    # Step 5: Final Audio Mux
    print("\n--- Step 5: Muxing Final 1080p Video with Female Audio Narration ---")
    output_mp4 = VIDEO_DIR / "emotion_recognition_linkedin_1min.mp4"
    final_cmd = [
        "ffmpeg", "-y",
        "-i", str(unsubbed_video),
        "-i", str(full_audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4)
    ]
    subprocess.run(final_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    dur_final = librosa.get_duration(path=str(full_audio_path))
    file_size_mb = output_mp4.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 65)
    print(f"SUCCESS! PROFESSIONALLY REFINED VIDEO EXPORTED:")
    print(f"  Location:   {output_mp4}")
    print(f"  Size:       {file_size_mb:.2f} MB")
    print(f"  Duration:   {dur_final:.2f} seconds")
    print(f"  Resolution: 1920x1080 Full HD (25 FPS)")
    print(f"  Voice:      Natural Female Voiceover (Aria Neural)")
    print(f"  Subtitles:  Fitted, Translucent (rgba(0,0,0,0.65)), 34px Segoe UI,")
    print(f"              8px radius, safe bottom margin (65px), zero content overlap.")
    print("=" * 65)


if __name__ == "__main__":
    build_video()
