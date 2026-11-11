# 🚀 LinkedIn Post Copy (Ready to Paste)

**Title**: Breaking Acoustic Memorization: Building a Speaker-Independent Speech Emotion Recognition (SER) System in PyTorch

---

🎙️ **Can AI distinguish human emotion from vocal timbre, or is it just memorizing who is talking?**

In Speech Emotion Recognition, random train-test splitting is the biggest trap in machine learning. Because an actor's vocal tract and pitch profile remain distinct across utterances, a model evaluated on random splits achieves deceptively high accuracy (~85%+) by simply identifying *the speaker*, not the *emotion*.

I built a production-grade Speech Emotion Recognition system addressing this exact failure mode:

### 🔍 Key Engineering Highlights:
1. **Strict Speaker-Independent Partitioning**: Partitioned 1,440 RAVDESS speech clips by Actor ID. Actors 21 to 24 were strictly quarantined for unseen out-of-sample testing with an exact 50/50 gender balance.
2. **280-Dimensional Acoustic Feature Pipeline**: Extracted 40 MFCCs, first-order velocity deltas, acceleration delta-deltas, 12 chroma bins, 7 spectral contrast bands, and RMS energy across 22,050 Hz normalized audio.
3. **Deep 1D PyTorch CNN with Weight Balancing**: Designed with 1D convolutions, batch normalization, spatial dropout (0.3), and class-weighted cross-entropy loss to prevent majority-class collapse.
4. **Honest Acoustic Confusion Matrix**: On completely unseen speakers, the model achieves **38.3% accuracy and 37.8% Macro F1** (3x over the 12.5% random baseline). Error analysis shows intuitive human acoustic overlaps: *Happy* vs *Angry* (shared high vocal arousal), and *Sadness* vs *Calm* (shared subdued fundamental frequency).
5. **Interactive Responsive Web App**: Luxury warm-espresso & cashmere UI with light/dark theme, real-time 3-second microphone recording with live waveform visualizer, audio player controls (Play, Pause, Stop, Delete), and a local recordings history deck.

📊 **Stack**: PyTorch | Librosa | ONNX Runtime | HTML5 Web Audio API

Check out the 1-minute video walk-through below! 👇

#MachineLearning #DeepLearning #SpeechEmotionRecognition #PyTorch #AudioProcessing #AI #DataScience #WebAudio #PortfolioProject
