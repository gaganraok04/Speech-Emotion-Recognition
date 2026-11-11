"""
LinkedIn 1-Minute Showcase Video Generator for Speech Emotion Recognition.

Features:
- Professional 1080p (1920x1080) slide design in warm brown espresso palette.
- Natural, expressive female voice narration using edge-tts (en-US-AriaNeural).
- Welcome slide & Thank You slide.
- Real-time working project demo scenes highlighting live 3s recording, waveform canvas, controls, and history.
- Synchronized subtitles (SRT) burned directly into the video for mobile and LinkedIn feed accessibility.
- Stitched using FFmpeg into a 60-second Full HD MP4 at 25 FPS.
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import edge_tts
import librosa

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "video"
SLIDES_DIR = VIDEO_DIR / "slides"
AUDIO_DIR = VIDEO_DIR / "audio"
CLIPS_DIR = VIDEO_DIR / "clips"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Fonts
FONT_REGULAR = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD = "C:/Windows/Fonts/segoeuib.ttf"
FONT_CODE = "C:/Windows/Fonts/consola.ttf"

# Palettes (Warm Brown & Espresso)
COLOR_BG = (18, 14, 11)           # #120e0b
COLOR_CARD = (34, 27, 21)         # #221b15
COLOR_BORDER = (217, 119, 6, 40)  # amber border
COLOR_AMBER = (217, 119, 6)       # #d97706
COLOR_AMBER_BRIGHT = (245, 158, 11)# #f59e0b
COLOR_TERRACOTTA = (194, 65, 12)  # #c2410c
COLOR_TEXT_MAIN = (252, 249, 245) # #fcf9f5
COLOR_TEXT_MUTED = (184, 168, 152)# #b8a898
COLOR_TEXT_DIM = (132, 114, 98)   # #847262
COLOR_RED = (220, 38, 38)
COLOR_GREEN = (22, 163, 74)

VOICE_FEMALE = "en-US-AriaNeural"

# Tightened 60-second script for LinkedIn Spotlight
SCENES = [
    {
        "id": "scene_01_welcome",
        "title": "Welcome Slide",
        "text": "Can an AI detect your true emotion just from the tone of your voice? Most speech models cheat by memorizing speaker voices. Here is how we solved it with PyTorch.",
        "duration_target": 9.5
    },
    {
        "id": "scene_02_speaker_split",
        "title": "The Golden Rule: Speaker Independence",
        "text": "If you split speech data at random, your accuracy is fiction. We enforced a strict speaker-independent split on twenty-four actors, testing only on completely unseen voices.",
        "duration_target": 10.5
    },
    {
        "id": "scene_03_features_model",
        "title": "280-D Acoustic Features & PyTorch CNN",
        "text": "We engineered a 280-dimensional acoustic feature vector capturing 40 MFCC bands, pitch velocity, and spectral contrast, trained with a deep 1D Convolutional Neural Network.",
        "duration_target": 11.0
    },
    {
        "id": "scene_04_live_demo",
        "title": "Real-Time Working Project & UI",
        "text": "Here is the live project! Featuring live three-second microphone recording with animated waveforms, instant emotion prediction, and a recordings history deck.",
        "duration_target": 10.5
    },
    {
        "id": "scene_05_confusion_analysis",
        "title": "Honest Acoustic Confusion Analysis",
        "text": "We delivered honest benchmark metrics. High arousal overlaps between anger and joy, while subtle sadness shares acoustic prosody with calm speech.",
        "duration_target": 9.5
    },
    {
        "id": "scene_06_thank_you",
        "title": "Thank You & GitHub Deployment",
        "text": "Thank you for watching! The project is open source on GitHub and live on GitHub Pages. Star the repository, test your voice, and let's connect on LinkedIn!",
        "duration_target": 9.0
    }
]


def get_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def create_base_canvas():
    img = Image.new("RGB", (1920, 1080), COLOR_BG)
    draw = ImageDraw.Draw(img)

    for r in range(400, 0, -20):
        alpha = int(12 * (1 - r / 400))
        glow_color = (217, 119, 6, alpha)
        draw.ellipse([(100 - r, 80 - r), (100 + r, 80 + r)], fill=glow_color)

    # Top Brand Navbar
    brand_font = get_font(FONT_BOLD, 26)
    badge_font = get_font(FONT_BOLD, 18)
    draw.text((90, 55), "🎙️ SPEECH EMOTION RECOGNITION (SER)", font=brand_font, fill=COLOR_AMBER_BRIGHT)
    
    draw.rounded_rectangle([(1530, 50), (1830, 92)], radius=20, fill=(34, 27, 21), outline=COLOR_AMBER, width=1)
    draw.text((1560, 60), "★ 1-MIN SPOTLIGHT", font=badge_font, fill=COLOR_AMBER_BRIGHT)

    draw.line([(80, 1000), (1840, 1000)], fill=(50, 40, 32), width=1)
    footer_font = get_font(FONT_REGULAR, 20)
    draw.text((90, 1020), "Built with PyTorch & Librosa • Open Source on GitHub", font=footer_font, fill=COLOR_TEXT_DIM)
    draw.text((1550, 1020), "LinkedIn Showcase Video", font=footer_font, fill=COLOR_TEXT_DIM)

    return img, draw


def render_slide_welcome(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(80, 140), (1840, 960)], radius=24, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    badge_font = get_font(FONT_BOLD, 22)
    draw.rounded_rectangle([(760, 200), (1160, 250)], radius=25, fill=(50, 36, 26), outline=COLOR_AMBER, width=1)
    draw.text((795, 212), "🔴 PROJECT SPOTLIGHT", font=badge_font, fill=COLOR_AMBER_BRIGHT)

    title_font = get_font(FONT_BOLD, 68)
    draw.text((180, 310), "Emotion Recognition from Speech", font=title_font, fill=COLOR_TEXT_MAIN)
    
    sub_font = get_font(FONT_REGULAR, 34)
    draw.text((180, 410), "Why Most Speech Emotion Models Fail — And How We Solved It", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    pills = [
        ("🛡️ Zero Speaker Leakage", "Tested strictly on 100% unseen voices"),
        ("🎛️ 280-D MFCC Acoustic Features", "Pitch velocity, timbre & energy"),
        ("📱 Live Web App & Audio Deck", "Microphone recording & waveform player")
    ]
    pill_font_title = get_font(FONT_BOLD, 24)
    pill_font_desc = get_font(FONT_REGULAR, 18)

    for i, (p_title, p_desc) in enumerate(pills):
        x = 180 + i * 530
        draw.rounded_rectangle([(x, 540), (x + 490, 740)], radius=16, fill=(24, 18, 14), outline=COLOR_AMBER, width=1)
        draw.text((x + 30, 580), p_title, font=pill_font_title, fill=COLOR_TEXT_MAIN)
        draw.text((x + 30, 640), p_desc, font=pill_font_desc, fill=COLOR_TEXT_MUTED)

    np.random.seed(42)
    for w in range(0, 1480, 20):
        amp = np.sin(w / 60) * np.cos(w / 120) * 35 + np.random.normal(0, 8)
        draw.line([(180 + w, 840 - amp), (180 + w, 840 + amp)], fill=COLOR_AMBER_BRIGHT, width=3)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_speaker_split(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(80, 140), (1840, 960)], radius=24, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 52)
    draw.text((130, 180), "The Golden Rule: Speaker-Independent Split", font=title_font, fill=COLOR_TEXT_MAIN)

    tag_font = get_font(FONT_REGULAR, 26)
    draw.text((130, 255), "Split by speaker, not at random — otherwise your score is fiction.", font=tag_font, fill=COLOR_AMBER_BRIGHT)

    box_font_h = get_font(FONT_BOLD, 30)
    box_font_body = get_font(FONT_REGULAR, 22)

    # Left: The Trap
    draw.rounded_rectangle([(130, 340), (920, 880)], radius=18, fill=(38, 22, 22), outline=COLOR_RED, width=2)
    draw.text((170, 380), "❌ The Random Split Trap (Data Leakage)", font=box_font_h, fill=(239, 68, 68))
    draw.text((170, 450), "• Test set contains the SAME speakers heard in training", font=box_font_body, fill=COLOR_TEXT_MAIN)
    draw.text((170, 510), "• Model memorizes actor vocal tract, timbre & pitch range", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((170, 570), "• Reports inflated '90-95%' test accuracy in papers", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((170, 630), "• COMPLETELY FAILS when deployed on any new human voice", font=box_font_body, fill=(239, 68, 68))
    draw.rounded_rectangle([(170, 740), (880, 820)], radius=12, fill=(60, 20, 20))
    draw.text((210, 765), "FICTITIOUS RESULT: Overfitting to Speaker Identity", font=get_font(FONT_BOLD, 22), fill=(255, 120, 120))

    # Right: Our Approach
    draw.rounded_rectangle([(980, 340), (1770, 880)], radius=18, fill=(24, 34, 24), outline=COLOR_GREEN, width=2)
    draw.text((1020, 380), "✅ Our Speaker-Independent Split (Reality)", font=box_font_h, fill=(34, 197, 94))
    draw.text((1020, 450), "• RAVDESS 24 professional actors partitioned strictly by ID", font=box_font_body, fill=COLOR_TEXT_MAIN)
    draw.text((1020, 510), "• Train: Actors 01-16 (8 Male, 8 Female, 66.7%)", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((1020, 570), "• Validation: Actors 17-20 (2 Male, 2 Female, 16.7%)", font=box_font_body, fill=COLOR_TEXT_MUTED)
    draw.text((1020, 630), "• Test: Actors 21-24 (2 Male, 2 Female) — 100% UNSEEN", font=box_font_body, fill=(34, 197, 94))
    draw.rounded_rectangle([(1020, 740), (1730, 820)], radius=12, fill=(20, 55, 25))
    draw.text((1060, 765), "HONEST RESULT: Generalizes to Real Human Emotions", font=get_font(FONT_BOLD, 22), fill=(150, 255, 160))

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_features_model(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(80, 140), (1840, 960)], radius=24, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 52)
    draw.text((130, 180), "280-Dimensional Acoustic Feature Pipeline", font=title_font, fill=COLOR_TEXT_MAIN)
    sub_font = get_font(FONT_REGULAR, 26)
    draw.text((130, 255), "Extracting spectral envelopes, velocity derivatives, and harmonic cues with Librosa", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    feats = [
        ("40 MFCCs (80-D)", "Mean & standard deviation across 40 Mel filterbanks mimicking non-linear human cochlear pitch perception.", COLOR_AMBER),
        ("Delta & Delta-Delta (160-D)", "1st & 2nd temporal derivatives capturing vocal pitch velocity, speech cadence, and sudden prosodic acceleration.", COLOR_TERRACOTTA),
        ("Chroma & Contrast (38-D)", "12 pitch chroma classes for intonation plus 7 frequency sub-band spectral contrast metrics.", COLOR_AMBER_BRIGHT),
        ("RMS Energy (2-D)", "Frame-level Root-Mean-Square energy tracking loudness contour and volume dynamic range.", (168, 153, 138))
    ]

    h_font = get_font(FONT_BOLD, 26)
    b_font = get_font(FONT_REGULAR, 20)

    for i, (f_name, f_desc, f_col) in enumerate(feats):
        y = 330 + i * 135
        draw.rounded_rectangle([(130, y), (1200, y + 115)], radius=14, fill=(24, 18, 14), outline=f_col, width=1)
        draw.text((160, y + 20), f_name, font=h_font, fill=f_col)
        draw.text((160, y + 65), f_desc, font=b_font, fill=COLOR_TEXT_MUTED)

    draw.rounded_rectangle([(1240, 330), (1770, 870)], radius=18, fill=(28, 21, 16), outline=COLOR_AMBER, width=2)
    draw.text((1280, 370), "🧠 PyTorch 1D CNN Architecture", font=get_font(FONT_BOLD, 26), fill=COLOR_TEXT_MAIN)
    
    arch_lines = [
        "• Conv1D Block 1: 64 filters (k=5)",
        "• Conv1D Block 2: 128 filters (k=5)",
        "• Conv1D Block 3: 256 filters (k=3)",
        "• Conv1D Block 4: 256 filters (k=3)",
        "• Batch Normalization & LeakyReLU",
        "• Dropout Regularization (0.25 - 0.35)",
        "• AdaptiveAvgPool1d Global Pooling",
        "• Dense Classifier: 256 → 128 → 8",
        "• Class-Weighted CrossEntropyLoss"
    ]
    for idx, line in enumerate(arch_lines):
        draw.text((1280, 430 + idx * 45), line, font=get_font(FONT_REGULAR, 20), fill=COLOR_AMBER_BRIGHT if "Conv" in line or "Class" in line else COLOR_TEXT_MUTED)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_live_demo(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(80, 140), (1840, 960)], radius=24, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 52)
    draw.text((130, 180), "Real-Time Working Project: Interactive Web UI", font=title_font, fill=COLOR_TEXT_MAIN)
    sub_font = get_font(FONT_REGULAR, 26)
    draw.text((130, 255), "Live 3-Second Microphone Recording • Audio Player Toolbar • Recordings History", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    draw.rounded_rectangle([(130, 330), (1100, 880)], radius=18, fill=(24, 18, 14), outline=COLOR_AMBER, width=2)
    
    draw.rounded_rectangle([(160, 360), (1070, 430)], radius=10, fill=(60, 20, 20), outline=COLOR_RED, width=1)
    draw.ellipse([(190, 385), (210, 405)], fill=COLOR_RED)
    draw.text((230, 380), "LIVE RECORDING ACTIVE — Speak into microphone...", font=get_font(FONT_BOLD, 20), fill=(255, 140, 140))
    draw.text((950, 375), "2.4s", font=get_font(FONT_BOLD, 32), fill=COLOR_RED)

    draw.rounded_rectangle([(160, 455), (1070, 585)], radius=10, fill=(18, 13, 10), outline=(50, 38, 30), width=1)
    np.random.seed(101)
    for w in range(0, 900, 10):
        amp = np.sin(w / 30) * np.sin(w / 80) * 45 + np.random.normal(0, 8)
        draw.line([(170 + w, 520 - amp), (170 + w, 520 + amp)], fill=COLOR_AMBER_BRIGHT, width=2)

    btns = [("▶ Play", (217, 119, 6)), ("⏸ Pause", (50, 38, 30)), ("⏹ Stop", (50, 38, 30)), ("🗑️ Delete", (120, 30, 30))]
    for b_idx, (b_label, b_color) in enumerate(btns):
        bx = 160 + b_idx * 230
        draw.rounded_rectangle([(bx, 610), (bx + 215, 665)], radius=8, fill=b_color)
        draw.text((bx + 60, 625), b_label, font=get_font(FONT_BOLD, 20), fill=COLOR_TEXT_MAIN)

    draw.rounded_rectangle([(160, 690), (1070, 850)], radius=12, fill=(32, 24, 18), outline=COLOR_AMBER, width=1)
    draw.text((190, 715), "CLASSIFIED EMOTION", font=get_font(FONT_BOLD, 18), fill=COLOR_TEXT_DIM)
    draw.text((190, 745), "ANGRY", font=get_font(FONT_BOLD, 46), fill=COLOR_AMBER_BRIGHT)
    draw.rounded_rectangle([(400, 755), (550, 795)], radius=15, fill=(80, 25, 25), outline=COLOR_RED, width=1)
    draw.text((425, 765), "High Arousal", font=get_font(FONT_BOLD, 16), fill=(255, 120, 120))
    
    draw.rounded_rectangle([(190, 810), (1040, 825)], radius=8, fill=(20, 15, 12))
    draw.rounded_rectangle([(190, 810), (190 + int(850 * 0.742), 825)], radius=8, fill=COLOR_AMBER_BRIGHT)

    draw.rounded_rectangle([(1140, 330), (1770, 880)], radius=18, fill=(28, 21, 16), outline=(60, 46, 36), width=2)
    draw.text((1180, 370), "📜 Live Recordings History Deck", font=get_font(FONT_BOLD, 26), fill=COLOR_TEXT_MAIN)

    hist_items = [
        ("Microphone Recording #3", "Just now • 3.00s", "Angry (74.2%)", COLOR_RED),
        ("Microphone Recording #2", "1 min ago • 3.00s", "Surprised (65.4%)", COLOR_AMBER_BRIGHT),
        ("Microphone Recording #1", "3 mins ago • 3.00s", "Calm (58.1%)", COLOR_GREEN),
        ("Actor 22 Test Sample", "5 mins ago • 3.20s", "Happy (60.0%)", COLOR_AMBER)
    ]
    for h_i, (h_title, h_sub, h_tag, h_col) in enumerate(hist_items):
        hy = 430 + h_i * 105
        draw.rounded_rectangle([(1180, hy), (1730, hy + 90)], radius=10, fill=(20, 15, 12), outline=(50, 38, 30), width=1)
        draw.text((1205, hy + 18), h_title, font=get_font(FONT_BOLD, 20), fill=COLOR_TEXT_MAIN)
        draw.text((1205, hy + 50), h_sub, font=get_font(FONT_REGULAR, 16), fill=COLOR_TEXT_DIM)
        draw.rounded_rectangle([(1520, hy + 25), (1705, hy + 65)], radius=12, fill=(40, 30, 24), outline=h_col, width=1)
        draw.text((1535, hy + 33), h_tag, font=get_font(FONT_BOLD, 15), fill=h_col)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_confusion(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(80, 140), (1840, 960)], radius=24, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 52)
    draw.text((130, 180), "Honest Acoustic Evaluation & Confusion Analysis", font=title_font, fill=COLOR_TEXT_MAIN)
    sub_font = get_font(FONT_REGULAR, 26)
    draw.text((130, 255), "Evaluated strictly on unseen Actors 21–24 (50% Male / 50% Female)", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    cm_img_path = REPORTS_DIR / "confusion_matrix.png"
    if cm_img_path.exists():
        cm_img = Image.open(cm_img_path).convert("RGB")
        cm_img = cm_img.resize((680, 530), Image.Resampling.LANCZOS)
        img.paste(cm_img, (130, 340))
    else:
        draw.rounded_rectangle([(130, 340), (810, 870)], radius=14, fill=(24, 18, 14))

    draw.rounded_rectangle([(850, 340), (1770, 870)], radius=18, fill=(28, 21, 16), outline=COLOR_AMBER, width=1)
    draw.text((890, 380), "🔍 What Does the Model Confuse and Why?", font=get_font(FONT_BOLD, 28), fill=COLOR_TEXT_MAIN)

    reasons = [
        ("High Arousal Overlap: Angry ↔ Happy", "Both share identical spectral signatures: elevated pitch (F0), rapid speech cadence, and sharp high-frequency energy. Without linguistic transcript, audio cannot distinguish aggressive vs joyous arousal."),
        ("Low Arousal Compression: Sad ↔ Neutral ↔ Calm", "Subdued dynamic range, slow tempo, and flat pitch contours make subtle sadness and calm speech acoustic cousins. Human listeners rely on facial expressions to distinguish them."),
        ("Jitter & Instability: Fear ↔ Surprise", "Both display sudden pitch excursions, vocal breathiness, and acoustic jitter.")
    ]

    r_title_font = get_font(FONT_BOLD, 22)
    r_body_font = get_font(FONT_REGULAR, 18)

    for idx, (rt, rb) in enumerate(reasons):
        ry = 450 + idx * 135
        draw.rounded_rectangle([(890, ry), (1730, ry + 115)], radius=10, fill=(20, 15, 12), outline=(50, 38, 30), width=1)
        draw.text((915, ry + 15), rt, font=r_title_font, fill=COLOR_AMBER_BRIGHT)
        words = rb.split()
        line1 = " ".join(words[:14])
        line2 = " ".join(words[14:])
        draw.text((915, ry + 50), line1, font=r_body_font, fill=COLOR_TEXT_MUTED)
        draw.text((915, ry + 78), line2, font=r_body_font, fill=COLOR_TEXT_MUTED)

    img.save(path)
    print(f"Rendered: {path.name}")


def render_slide_thank_you(path: Path):
    img, draw = create_base_canvas()
    draw.rounded_rectangle([(80, 140), (1840, 960)], radius=24, fill=COLOR_CARD, outline=(60, 46, 36), width=2)

    title_font = get_font(FONT_BOLD, 68)
    draw.text((180, 260), "Thank You for Watching!", font=title_font, fill=COLOR_TEXT_MAIN)
    sub_font = get_font(FONT_REGULAR, 32)
    draw.text((180, 360), "Speech Emotion Recognition (SER) Project is Open Source", font=sub_font, fill=COLOR_AMBER_BRIGHT)

    ctas = [
        ("⭐ Star on GitHub", "Explore clean PyTorch code, datasets, & training pipelines.", COLOR_AMBER),
        ("🌐 Try Live Web Demo", "Instant deployment on GitHub Pages with live microphone recording.", COLOR_TERRACOTTA),
        ("🤝 Connect on LinkedIn", "Let's discuss machine learning, speech processing, and audio AI.", COLOR_GREEN)
    ]
    c_title = get_font(FONT_BOLD, 26)
    c_desc = get_font(FONT_REGULAR, 19)

    for i, (ct, cd, cc) in enumerate(ctas):
        x = 180 + i * 530
        draw.rounded_rectangle([(x, 480), (x + 490, 720)], radius=16, fill=(24, 18, 14), outline=cc, width=2)
        draw.text((x + 35, 520), ct, font=c_title, fill=cc)
        draw.text((x + 35, 590), cd[:36], font=c_desc, fill=COLOR_TEXT_MAIN)
        draw.text((x + 35, 625), cd[36:], font=c_desc, fill=COLOR_TEXT_MUTED)

    techs = ["PyTorch 2.0", "Librosa Audio", "RAVDESS Benchmark", "HTML5 Web Audio", "GitHub Actions"]
    t_font = get_font(FONT_BOLD, 18)
    for t_i, tech in enumerate(techs):
        tx = 180 + t_i * 300
        draw.rounded_rectangle([(tx, 790), (tx + 270, 840)], radius=25, fill=(38, 28, 22), outline=COLOR_AMBER, width=1)
        draw.text((tx + 30, 804), f"✓ {tech}", font=t_font, fill=COLOR_AMBER_BRIGHT)

    img.save(path)
    print(f"Rendered: {path.name}")


async def generate_speech(text: str, output_path: Path):
    communicate = edge_tts.Communicate(text, VOICE_FEMALE, rate="+8%")
    await communicate.save(str(output_path))


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

    print("\n--- Step 1: Rendering High-Resolution 1080p Video Slides ---")
    render_slide_welcome(SLIDES_DIR / "slide_01.png")
    render_slide_speaker_split(SLIDES_DIR / "slide_02.png")
    render_slide_features_model(SLIDES_DIR / "slide_03.png")
    render_slide_live_demo(SLIDES_DIR / "slide_04.png")
    render_slide_confusion(SLIDES_DIR / "slide_05.png")
    render_slide_thank_you(SLIDES_DIR / "slide_06.png")

    print("\n--- Step 2: Generating Female Voice Narration (edge-tts) ---")
    audio_durations = []
    srt_entries = []
    current_time = 0.0

    for idx, scene in enumerate(SCENES):
        audio_file = AUDIO_DIR / f"{scene['id']}.mp3"
        print(f"Synthesizing voice for {scene['title']}...")
        asyncio.run(generate_speech(scene["text"], audio_file))

        dur = librosa.get_duration(path=str(audio_file))
        padded_dur = round(dur + 0.35, 2)
        audio_durations.append(padded_dur)

        words = scene["text"].split()
        mid = len(words) // 2
        part1 = " ".join(words[:mid])
        part2 = " ".join(words[mid:])

        half_dur = padded_dur / 2.0

        srt_entries.append({
            "idx": len(srt_entries) + 1,
            "start": format_srt_time(current_time),
            "end": format_srt_time(current_time + half_dur),
            "text": part1
        })
        srt_entries.append({
            "idx": len(srt_entries) + 1,
            "start": format_srt_time(current_time + half_dur),
            "end": format_srt_time(current_time + padded_dur),
            "text": part2
        })

        current_time += padded_dur

    total_duration = current_time
    print(f"\nTotal Video Duration: {total_duration:.1f} seconds (Target: ~60s)")

    srt_path = VIDEO_DIR / "subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for entry in srt_entries:
            f.write(f"{entry['idx']}\n")
            f.write(f"{entry['start']} --> {entry['end']}\n")
            f.write(f"{entry['text']}\n\n")
    print(f"Subtitles written to {srt_path}")

    # Step 3: Stitching Audio Narration
    print("\n--- Step 3: Stitching Audio Narration ---")
    concat_audio_txt = AUDIO_DIR / "concat_audio.txt"
    with open(concat_audio_txt, "w", encoding="utf-8") as f:
        for scene in SCENES:
            f.write(f"file '{scene['id']}.mp3'\n")

    full_audio_path = VIDEO_DIR / "full_narration.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_audio_txt),
        "-c", "copy",
        str(full_audio_path)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Full audio narration saved to {full_audio_path}")

    # Step 4: Render Constant-Frame-Rate (25 fps) Video Clips for Each Slide
    print("\n--- Step 4: Rendering Constant-Frame-Rate Clips for Each Slide ---")
    clip_list_txt = CLIPS_DIR / "clip_list.txt"
    clip_paths = []

    with open(clip_list_txt, "w", encoding="utf-8") as f:
        for idx, (scene, dur) in enumerate(zip(SCENES, audio_durations)):
            slide_png = SLIDES_DIR / f"slide_0{idx+1}.png"
            clip_mp4 = CLIPS_DIR / f"clip_0{idx+1}.mp4"
            
            # Create a 25fps constant video clip matching the exact duration
            cmd_clip = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-i", str(slide_png),
                "-t", str(dur),
                "-r", "25",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                str(clip_mp4)
            ]
            subprocess.run(cmd_clip, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f.write(f"file '{clip_mp4.resolve().as_posix()}'\n")
            clip_paths.append(clip_mp4)
            print(f"Generated 25fps clip for Slide {idx+1} ({dur:.1f}s)")

    # Step 5: Concatenate clips without audio
    print("\n--- Step 5: Concatenating Slides & Merging Audio with Burned Subtitles ---")
    unsubbed_video = VIDEO_DIR / "unsubbed_video.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(clip_list_txt),
        "-c", "copy",
        str(unsubbed_video)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Step 6: Final Mux with Audio & Burn-in Subtitles
    output_mp4 = VIDEO_DIR / "emotion_recognition_linkedin_1min.mp4"
    escaped_srt = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")
    subtitle_filter = (
        f"subtitles='{escaped_srt}':force_style='FontName=Segoe UI,FontSize=22,Bold=1,"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=1,"
        f"BackColour=&H80120E0B,MarginV=35'"
    )

    final_cmd = [
        "ffmpeg", "-y",
        "-i", str(unsubbed_video),
        "-i", str(full_audio_path),
        "-vf", subtitle_filter,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_mp4)
    ]
    print("Muxing final video with subtitles...")
    subprocess.run(final_cmd, check=True)

    print("\n" + "=" * 65)
    print(f"🎉 SUCCESS! 1-MINUTE LINKEDIN VIDEO READY:")
    print(f"  Location: {output_mp4}")
    print(f"  Size: {output_mp4.stat().st_size / (1024*1024):.2f} MB")
    print(f"  Resolution: 1920x1080 Full HD (25 FPS)")
    print(f"  Voice: Natural Female Voiceover (Aria Neural)")
    print(f"  Subtitles: Burned-in, mobile-optimized captions")
    print("=" * 65)


if __name__ == "__main__":
    build_video()
