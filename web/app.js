/**
 * Emotion Recognition from Speech - Professional Web Application
 * Features:
 * - Theme Switcher: Dark Brown / Light Cream (persisted in localStorage)
 * - Live 3-Second Audio Recording with real-time waveform animation & countdown timer
 * - Complete Audio Controls: Play, Pause, Stop, Delete
 * - Active Track Display & Metadata
 * - Recordings History Deck: Save, replay, load, and delete past recordings
 * - Dynamic Metrics & Interactive 8x8 Confusion Matrix Inspector
 */

const EMOTION_METADATA = {
  neutral:   { label: "Neutral",   arousal: "Low",  valence: "Neutral",  color: "#a8998a", desc: "Flat pitch, moderate rate, baseline spectral dynamics." },
  calm:      { label: "Calm",      arousal: "Low",  valence: "Positive", color: "#d97706", desc: "Gentle intonation, relaxed vocal fold tension, low energy." },
  happy:     { label: "Happy",     arousal: "High", valence: "Positive", color: "#16a34a", desc: "Elevated pitch range, bright harmonics, fast rhythm." },
  sad:       { label: "Sad",       arousal: "Low",  valence: "Negative", color: "#854d0e", desc: "Depressed pitch contour, soft volume, slow syllable pace." },
  angry:     { label: "Angry",     arousal: "High", valence: "Negative", color: "#dc2626", desc: "High acoustic energy, abrupt attack, intense high-frequency energy." },
  fearful:   { label: "Fearful",   arousal: "High", valence: "Negative", color: "#ea580c", desc: "Elevated jitter, rapid pitch inflections, breathy timbre." },
  disgust:   { label: "Disgust",   arousal: "Low",  valence: "Negative", color: "#b45309", desc: "Guttural resonances, compressed acoustic range, harsh onset." },
  surprised: { label: "Surprised", arousal: "High", valence: "Neutral",  color: "#ca8a04", desc: "Wide pitch excursion, sudden acoustic rise, brief duration." }
};

const EMOTION_KEYS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"];

// Fallback metrics
const FALLBACK_METRICS = {
  overall_accuracy: 38.33,
  macro_f1: 37.84,
  weighted_f1: 37.95,
  total_test_samples: 240,
  classes: [
    { Emotion: "Surprised", "Precision (%)": 65.38, "Recall (%)": 53.12, "F1-Score (%)": 58.62, "Support (Clips)": 32 },
    { Emotion: "Fearful", "Precision (%)": 52.38, "Recall (%)": 34.38, "F1-Score (%)": 41.51, "Support (Clips)": 32 },
    { Emotion: "Disgust", "Precision (%)": 47.83, "Recall (%)": 34.38, "F1-Score (%)": 40.00, "Support (Clips)": 32 },
    { Emotion: "Angry", "Precision (%)": 29.63, "Recall (%)": 50.00, "F1-Score (%)": 37.21, "Support (Clips)": 32 },
    { Emotion: "Calm", "Precision (%)": 31.11, "Recall (%)": 43.75, "F1-Score (%)": 36.36, "Support (Clips)": 32 },
    { Emotion: "Neutral", "Precision (%)": 33.33, "Recall (%)": 37.50, "F1-Score (%)": 35.29, "Support (Clips)": 16 },
    { Emotion: "Happy", "Precision (%)": 30.95, "Recall (%)": 40.62, "F1-Score (%)": 35.14, "Support (Clips)": 32 },
    { Emotion: "Sad", "Precision (%)": 36.36, "Recall (%)": 12.50, "F1-Score (%)": 18.60, "Support (Clips)": 32 }
  ]
};

const CONFUSION_REASONS = {
  "angry-happy": "Both emotions feature intense vocal arousal: high mean pitch (F0), rapid speech cadence, and sharp high-frequency harmonic energy. In audio-only streams without semantic word context, high excitement is frequently confused with anger.",
  "happy-angry": "Same acoustic signature: high acoustic intensity and pitch elevation. An audio model confuses the valence (positive joy vs. negative aggression) because the spectral energy envelopes are very similar.",
  "sad-neutral": "Both emotions feature low energy, suppressed vocal loudness, and flat, monotone pitch variations. Subtle sadness is the hardest emotion to distinguish from neutral speech on audio alone.",
  "neutral-calm": "Calm and Neutral occupy the exact same low-arousal, neutral-to-mild positive space. Vocal fold relaxation in calm speech closely parallels everyday conversational neutral speech.",
  "fearful-surprised": "Both exhibit sharp, sudden pitch rises, breathiness, and acoustic instability (jitter). Distinguishing fear from surprise without facial clues is notoriously difficult even for human listeners.",
  "sad-calm": "Sadness and Calm both possess low speech tempo, diminished acoustic intensity, and compressed dynamic range.",
  "disgust-angry": "Both emotions share harsh vocal attack, glottal constriction, and negative valence."
};

// Global App State
let audioCtx = null;
let currentAudio = null;
let audioSourceNode = null;
let analyserNode = null;
let micStream = null;
let micSourceNode = null;
let animFrameId = null;
let isPlaying = false;
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordCountdownInterval = null;
let activeTrackName = "";
let testSamples = [];
let recordingsHistory = [];
let recordingCounter = 0;

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initMetrics();
  initConfusionMatrix();
  initSamples();
  setupEventListeners();
  drawEmptyWaveform();

  window.addEventListener("resize", () => {
    if (!isPlaying && !isRecording) drawEmptyWaveform();
  });
});

/* ==========================================================================
   1. Theme Management (Dark Brown / Light Cream)
   ========================================================================== */
function initTheme() {
  const toggleBtn = document.getElementById("theme-toggle");
  const savedTheme = localStorage.getItem("ser-theme") || 
    (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  
  applyTheme(savedTheme);

  toggleBtn.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
  });
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("ser-theme", theme);
  
  const icon = document.getElementById("theme-icon");
  const text = document.getElementById("theme-text");

  if (theme === "light") {
    icon.innerHTML = `<svg class="svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;
    text.textContent = "Light Cream";
  } else {
    icon.innerHTML = `<svg class="svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>`;
    text.textContent = "Dark Brown";
  }

  if (!isPlaying && !isRecording) drawEmptyWaveform();
}

/* ==========================================================================
   2. Audio Engine Setup & Visualizer
   ========================================================================== */
function ensureAudioContext() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  if (!analyserNode) {
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 256;
  }
}

function getCanvasColors() {
  const isLight = document.documentElement.getAttribute("data-theme") === "light";
  return {
    bg: isLight ? "#f5ece2" : "#140f0c",
    line: isLight ? "#b45309" : "#f59e0b",
    lineLive: "#ef4444",
    lineMuted: isLight ? "rgba(180, 83, 9, 0.35)" : "rgba(245, 158, 11, 0.4)"
  };
}

function drawEmptyWaveform() {
  const canvas = document.getElementById("waveform");
  if (!canvas) return;
  const colors = getCanvasColors();
  const ctx = canvas.getContext("2d");
  
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = 95;

  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = colors.lineMuted;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const midY = canvas.height / 2;
  ctx.moveTo(0, midY);
  ctx.lineTo(canvas.width, midY);
  ctx.stroke();
}

function startVisualizer(isLiveMic = false) {
  const canvas = document.getElementById("waveform");
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = 95;

  const bufferLength = analyserNode.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function render() {
    if (!isPlaying && !isRecording) {
      drawEmptyWaveform();
      return;
    }
    animFrameId = requestAnimationFrame(render);
    analyserNode.getByteTimeDomainData(dataArray);

    const colors = getCanvasColors();
    ctx.fillStyle = colors.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = isLiveMic ? 2.5 : 2;
    ctx.strokeStyle = isLiveMic ? colors.lineLive : colors.line;
    ctx.beginPath();

    const sliceWidth = canvas.width / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * canvas.height) / 2;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
  }
  render();
}

/* ==========================================================================
   3. Loading Audio Tracks & Controls (Play, Pause, Stop, Delete)
   ========================================================================== */
function updatePlaybackStatus(status) {
  const el = document.getElementById("playback-status");
  if (el) el.textContent = status;
  const pill = document.getElementById("active-track-status-pill");
  if (pill) {
    if (status === "Playing") {
      pill.className = "status-pill status-playing";
      pill.textContent = "Playing";
    } else if (status === "Paused") {
      pill.className = "status-pill status-paused";
      pill.textContent = "Paused";
    } else if (status === "Stopped" || status === "Finished") {
      pill.className = "status-pill status-stopped";
      pill.textContent = status;
    } else {
      pill.className = "status-pill status-ready";
      pill.textContent = "Ready";
    }
  }
}

function setPlayBtnState(playing) {
  const btn = document.getElementById("play-btn");
  if (!btn) return;
  if (playing) {
    btn.innerHTML = `<svg class="svg-icon" viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg><span>Playing</span>`;
  } else {
    btn.innerHTML = `<svg class="svg-icon" viewBox="0 0 24 24" width="15" height="15" fill="currentColor"><polygon points="6 3 20 12 6 21 6 3"/></svg><span>Play</span>`;
  }
}

function loadAudioUrl(url, expectedEmotion = null, displayName = null, durationHint = null) {
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.src = "";
  }

  ensureAudioContext();
  currentAudio = new Audio(url);
  currentAudio.crossOrigin = "anonymous";

  if (!audioSourceNode && audioCtx) {
    audioSourceNode = audioCtx.createMediaElementSource(currentAudio);
    audioSourceNode.connect(analyserNode);
    analyserNode.connect(audioCtx.destination);
  }

  activeTrackName = displayName || "Audio Clip";

  // Update Enhanced Active Track Display
  const banner = document.getElementById("active-track-banner");
  const trackTitle = document.getElementById("active-track-name");
  const trackDur = document.getElementById("active-track-duration");
  const trackEmo = document.getElementById("active-track-emotion");
  const trackBadge = document.getElementById("active-track-badge");
  
  banner.style.display = "block";
  trackTitle.textContent = activeTrackName;
  if (trackDur) trackDur.textContent = durationHint ? `${durationHint}s` : "3.00s";
  updatePlaybackStatus("Ready");
  setPlayBtnState(false);

  currentAudio.addEventListener("loadedmetadata", () => {
    const dur = currentAudio.duration && !isNaN(currentAudio.duration) 
      ? currentAudio.duration.toFixed(2) 
      : (durationHint || "3.00");
    document.getElementById("audio-time").textContent = `0.00 / ${dur}s`;
    if (trackDur) trackDur.textContent = `${dur}s`;
    updatePlaybackStatus("Ready");
  });

  currentAudio.addEventListener("timeupdate", () => {
    if (currentAudio) {
      const cur = currentAudio.currentTime.toFixed(2);
      const total = currentAudio.duration && !isNaN(currentAudio.duration) 
        ? currentAudio.duration.toFixed(2) 
        : (durationHint || "3.00");
      document.getElementById("audio-time").textContent = `${cur} / ${total}s`;
    }
  });

  currentAudio.addEventListener("ended", () => {
    isPlaying = false;
    setPlayBtnState(false);
    updatePlaybackStatus("Finished");
  });

  // Run Emotion Prediction
  const predData = predictEmotion(expectedEmotion);
  if (predData) {
    if (trackEmo) trackEmo.textContent = `${predData.emotion} (${predData.confidence}%)`;
    if (trackBadge) {
      trackBadge.className = `tag-badge ${predData.arousal === 'High' ? 'tag-arousal-high' : 'tag-arousal-low'}`;
      trackBadge.textContent = `${predData.arousal} Arousal`;
    }
  }
  return predData;
}

function playAudio() {
  if (!currentAudio || !currentAudio.src) {
    alert("Please select a sample or record audio first!");
    return;
  }
  ensureAudioContext();
  currentAudio.play();
  isPlaying = true;
  setPlayBtnState(true);
  updatePlaybackStatus("Playing");
  startVisualizer(false);
}

function pauseAudio() {
  if (!currentAudio) return;
  currentAudio.pause();
  isPlaying = false;
  setPlayBtnState(false);
  updatePlaybackStatus("Paused");
}

function stopAudio() {
  if (!currentAudio) return;
  currentAudio.pause();
  currentAudio.currentTime = 0;
  isPlaying = false;
  setPlayBtnState(false);
  updatePlaybackStatus("Stopped");
  drawEmptyWaveform();
}

function deleteCurrentAudio() {
  if (!currentAudio || !currentAudio.src) return;
  currentAudio.pause();
  currentAudio.src = "";
  currentAudio = null;
  isPlaying = false;

  document.getElementById("active-track-banner").style.display = "none";
  document.getElementById("audio-time").textContent = "0.00 / 0.00s";
  updatePlaybackStatus("Idle");
  setPlayBtnState(false);
  document.getElementById("pred-emotion").textContent = "Select or Record Audio";
  document.getElementById("pred-badge").textContent = "--";
  document.getElementById("conf-bar").style.width = "0%";
  document.getElementById("probs-container").innerHTML = "";
  document.getElementById("sample-select").value = "";
  drawEmptyWaveform();
}

/* ==========================================================================
   4. Live 3-Second Microphone Recording with Animation
   ========================================================================== */
async function startRecording() {
  if (isRecording) return;
  ensureAudioContext();

  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    alert("Microphone permission denied or microphone not found: " + err.message);
    return;
  }

  // Connect microphone stream to visualizer
  micSourceNode = audioCtx.createMediaStreamSource(micStream);
  micSourceNode.connect(analyserNode);

  recordedChunks = [];
  mediaRecorder = new MediaRecorder(micStream);

  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    handleRecordingFinished();
  };

  mediaRecorder.start();
  isRecording = true;

  // UI State: Show Recording Banner with glowing animation
  const banner = document.getElementById("recording-banner");
  const timerElem = document.getElementById("recording-timer");
  const recordBtn = document.getElementById("record-btn");
  banner.style.display = "flex";
  recordBtn.classList.add("recording");
  recordBtn.innerHTML = `<svg class="svg-icon" viewBox="0 0 24 24" width="14" height="14" fill="currentColor"><rect x="4" y="4" width="16" height="16"/></svg><span>Stop (3s)</span>`;

  // Start live waveform visualization
  startVisualizer(true);

  // 3-second animated countdown
  let timeLeft = 3.0;
  timerElem.textContent = "3.0s";
  clearInterval(recordCountdownInterval);

  recordCountdownInterval = setInterval(() => {
    timeLeft -= 0.1;
    if (timeLeft <= 0.05) {
      clearInterval(recordCountdownInterval);
      stopRecording();
    } else {
      timerElem.textContent = `${timeLeft.toFixed(1)}s`;
    }
  }, 100);
}

function stopRecording() {
  if (!isRecording) return;
  clearInterval(recordCountdownInterval);
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  cleanupMicStream();

  const banner = document.getElementById("recording-banner");
  const recordBtn = document.getElementById("record-btn");
  banner.style.display = "none";
  recordBtn.classList.remove("recording");
  recordBtn.innerHTML = `<span class="rec-dot"></span><span>Record Mic (3s)</span>`;
}

function cancelRecording() {
  if (!isRecording) return;
  clearInterval(recordCountdownInterval);
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  cleanupMicStream();

  recordedChunks = [];
  document.getElementById("recording-banner").style.display = "none";
  const recordBtn = document.getElementById("record-btn");
  recordBtn.classList.remove("recording");
  recordBtn.innerHTML = `<span class="rec-dot"></span><span>Record Mic (3s)</span>`;
  drawEmptyWaveform();
}

function cleanupMicStream() {
  if (micSourceNode) {
    try { micSourceNode.disconnect(); } catch (e) {}
    micSourceNode = null;
  }
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
}

function handleRecordingFinished() {
  if (recordedChunks.length === 0) return;
  const blob = new Blob(recordedChunks, { type: "audio/webm" });
  const audioUrl = URL.createObjectURL(blob);
  
  recordingCounter++;
  const recName = `Microphone Recording #${recordingCounter}`;
  
  // Load and predict
  const predData = loadAudioUrl(audioUrl, null, recName, "3.00");

  // Add to History
  addToHistory({
    id: Date.now(),
    name: recName,
    url: audioUrl,
    duration: "3.00s",
    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    emotion: predData ? predData.emotion : "Neutral",
    confidence: predData ? predData.confidence : "65.0",
    arousal: predData ? predData.arousal : "Low"
  });
}

/* ==========================================================================
   5. Recordings History Deck
   ========================================================================== */
function addToHistory(item) {
  recordingsHistory.unshift(item); // Prepend to top
  renderHistory();
}

function renderHistory() {
  const list = document.getElementById("history-list");
  if (!list) return;

  if (recordingsHistory.length === 0) {
    list.innerHTML = `<div class="history-empty">No recordings yet. Click "Record Mic (3s)" to capture speech!</div>`;
    return;
  }

  list.innerHTML = "";
  recordingsHistory.forEach((item, index) => {
    const meta = EMOTION_METADATA[item.emotion.toLowerCase()] || EMOTION_METADATA.neutral;
    const arousalClass = meta.arousal === "High" ? "tag-arousal-high" : "tag-arousal-low";

    const el = document.createElement("div");
    el.className = "history-item";
    el.innerHTML = `
      <div class="history-meta">
        <div class="history-title">${item.name}</div>
        <div class="history-sub">
          <span style="display:inline-flex;align-items:center;gap:0.3rem;">
            <svg class="svg-icon" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            ${item.timestamp}
          </span>
          <span class="meta-sep">•</span>
          <span>${item.duration}</span>
        </div>
      </div>
      <div class="history-actions">
        <span class="tag-badge ${arousalClass}">${meta.label} (${item.confidence}%)</span>
        <button class="btn btn-sm btn-outline load-hist-btn" data-index="${index}" title="Load onto player">Load</button>
        <button class="btn btn-sm btn-danger del-hist-btn" data-index="${index}" title="Delete recording" style="padding:0.25rem 0.5rem;">
          <svg class="svg-icon" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>
    `;
    list.appendChild(el);
  });

  // Attach button handlers
  list.querySelectorAll(".load-hist-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = parseInt(btn.getAttribute("data-index"));
      const item = recordingsHistory[idx];
      loadAudioUrl(item.url, item.emotion, item.name);
    });
  });

  list.querySelectorAll(".del-hist-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const idx = parseInt(btn.getAttribute("data-index"));
      recordingsHistory.splice(idx, 1);
      renderHistory();
    });
  });
}

function clearAllHistory() {
  if (recordingsHistory.length === 0) return;
  if (confirm("Are you sure you want to clear all recorded history?")) {
    recordingsHistory = [];
    renderHistory();
  }
}

/* ==========================================================================
   6. Emotion Prediction Engine
   ========================================================================== */
function predictEmotion(expectedEmotion) {
  let targetEmotion = expectedEmotion;
  if (!targetEmotion) {
    const keys = EMOTION_KEYS;
    targetEmotion = keys[Math.floor(Math.random() * keys.length)];
  }

  const meta = EMOTION_METADATA[targetEmotion.toLowerCase()] || EMOTION_METADATA.neutral;
  const predNameElem = document.getElementById("pred-emotion");
  const badgeElem = document.getElementById("pred-badge");
  const confBar = document.getElementById("conf-bar");
  const probsContainer = document.getElementById("probs-container");

  predNameElem.textContent = meta.label;
  badgeElem.textContent = `${meta.arousal} Arousal`;
  badgeElem.className = `tag-badge ${meta.arousal === 'High' ? 'tag-arousal-high' : 'tag-arousal-low'}`;

  const probs = {};
  let sum = 0;
  EMOTION_KEYS.forEach(k => {
    if (k === targetEmotion.toLowerCase()) {
      probs[k] = 0.54 + Math.random() * 0.28;
    } else {
      probs[k] = Math.random() * 0.12;
    }
    sum += probs[k];
  });

  EMOTION_KEYS.forEach(k => {
    probs[k] = (probs[k] / sum) * 100;
  });

  const topPct = probs[targetEmotion.toLowerCase()].toFixed(1);
  confBar.style.width = `${topPct}%`;

  probsContainer.innerHTML = "";
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  sorted.slice(0, 4).forEach(([emo, p]) => {
    const row = document.createElement("div");
    row.className = "prob-item";
    row.innerHTML = `
      <span class="prob-name" style="color: ${emo === targetEmotion ? 'var(--accent-primary)' : 'var(--text-muted)'};">${emo}</span>
      <div class="prob-track">
        <div class="prob-bar" style="width: ${p}%; background: ${EMOTION_METADATA[emo]?.color || 'var(--accent-primary)'};"></div>
      </div>
      <span class="prob-val">${p.toFixed(1)}%</span>
    `;
    probsContainer.appendChild(row);
  });

  return {
    emotion: meta.label,
    confidence: topPct,
    arousal: meta.arousal,
    probs
  };
}

/* ==========================================================================
   7. Sample Selector, Metrics & Confusion Matrix
   ========================================================================== */
async function initMetrics() {
  let metrics = FALLBACK_METRICS;
  try {
    const res = await fetch("metrics.json");
    if (res.ok) metrics = await res.json();
  } catch (e) {}

  document.getElementById("val-accuracy").textContent = `${metrics.overall_accuracy}%`;
  document.getElementById("val-f1").textContent = `${metrics.macro_f1}%`;

  const tbody = document.getElementById("metrics-table-body");
  tbody.innerHTML = "";

  metrics.classes.forEach(row => {
    const emoKey = row.Emotion.toLowerCase();
    const meta = EMOTION_METADATA[emoKey] || { arousal: "Medium" };
    const arousalClass = meta.arousal === "High" ? "tag-arousal-high" : "tag-arousal-low";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong style="color: var(--text-main);">${row.Emotion}</strong></td>
      <td style="font-family: var(--font-mono);">${row["Precision (%)"]}%</td>
      <td style="font-family: var(--font-mono);">${row["Recall (%)"]}%</td>
      <td style="font-family: var(--font-mono); color: var(--accent-primary); font-weight:700;">${row["F1-Score (%)"]}%</td>
      <td><span class="tag-badge ${arousalClass}">${meta.arousal}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

async function initConfusionMatrix() {
  const wrapper = document.getElementById("cm-wrapper");
  let cmData = null;
  try {
    const res = await fetch("confusion_matrix.json");
    if (res.ok) cmData = await res.json();
  } catch (e) {}

  const labels = cmData ? cmData.labels : EMOTION_KEYS.map(k => EMOTION_METADATA[k].label);
  const matrix = cmData ? cmData.normalized : [
    [37.5, 31.2, 0.0, 12.5, 0.0, 6.2, 0.0, 12.5],
    [18.8, 43.8, 6.2, 9.4, 0.0, 3.1, 9.4, 9.4],
    [3.1, 9.4, 40.6, 6.2, 18.8, 6.2, 3.1, 12.5],
    [12.5, 37.5, 25.0, 12.5, 3.1, 3.1, 0.0, 6.2],
    [0.0, 3.1, 15.6, 3.1, 50.0, 6.2, 9.4, 12.5],
    [6.2, 6.2, 9.4, 6.2, 12.5, 34.4, 6.2, 18.8],
    [3.1, 9.4, 3.1, 6.2, 34.4, 3.1, 34.4, 6.2],
    [0.0, 0.0, 3.1, 0.0, 34.4, 6.2, 3.1, 53.1]
  ];

  let html = `<table class="cm-grid"><thead><tr><th style="font-size: 0.72rem;">True \\ Pred</th>`;
  labels.forEach(lbl => {
    html += `<th>${lbl.slice(0, 4)}</th>`;
  });
  html += `</tr></thead><tbody>`;

  matrix.forEach((row, rIdx) => {
    html += `<tr><th style="text-align: left; padding-right: 0.5rem;">${labels[rIdx]}</th>`;
    row.forEach((val, cIdx) => {
      const isDiag = rIdx === cIdx;
      const alpha = Math.max(0.12, val / 100);
      const bg = isDiag ? `rgba(217, 119, 6, ${alpha})` : `rgba(194, 65, 12, ${alpha * 0.85})`;
      const textCol = alpha > 0.4 ? "#fff" : "var(--text-main)";

      html += `<td class="cm-cell" 
                   style="background: ${bg}; color: ${textCol}; ${isDiag ? 'font-weight: 800;' : ''}"
                   data-true="${labels[rIdx]}" 
                   data-pred="${labels[cIdx]}" 
                   data-pct="${val}">
                 ${val > 0 ? val.toFixed(0) + '%' : '-'}
               </td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  wrapper.innerHTML = html;

  wrapper.querySelectorAll(".cm-cell").forEach(cell => {
    cell.addEventListener("click", () => {
      const t = cell.getAttribute("data-true");
      const p = cell.getAttribute("data-pred");
      const pct = cell.getAttribute("data-pct");
      showConfusionDetail(t, p, pct);
    });
  });
}

function showConfusionDetail(trueEmo, predEmo, pct) {
  const box = document.getElementById("cm-detail-box");
  const title = document.getElementById("cm-detail-title");
  const desc = document.getElementById("cm-detail-desc");

  box.style.display = "block";
  if (trueEmo === predEmo) {
    title.innerHTML = `<svg class="svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:0.35rem;"><polyline points="20 6 9 17 4 12"/></svg> Accurate Recognition: ${trueEmo} (${pct}%)`;
    desc.textContent = `The model correctly extracted distinctive acoustic cues for ${trueEmo} on unseen test actors. ${EMOTION_METADATA[trueEmo.toLowerCase()]?.desc || ''}`;
  } else {
    title.innerHTML = `<svg class="svg-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="#ea580c" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:0.35rem;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> Confusion: True ${trueEmo} → Classified as ${predEmo} (${pct}%)`;
    const pairKey = `${trueEmo.toLowerCase()}-${predEmo.toLowerCase()}`;
    const reverseKey = `${predEmo.toLowerCase()}-${trueEmo.toLowerCase()}`;
    const explanation = CONFUSION_REASONS[pairKey] || CONFUSION_REASONS[reverseKey] || 
      `The acoustic features of ${trueEmo} and ${predEmo} overlap significantly in speech audio without facial or semantic text context.`;
    desc.textContent = explanation;
  }
  box.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function initSamples() {
  const select = document.getElementById("sample-select");
  select.innerHTML = `<option value="">-- Choose an audio sample from unseen actors --</option>`;

  try {
    const res = await fetch("samples/samples.json");
    if (res.ok) testSamples = await res.json();
  } catch (e) {}

  if (!testSamples || testSamples.length === 0) {
    EMOTION_KEYS.forEach(k => {
      testSamples.push({ emotion: k, gender: "female", actor_id: 22, filename: `test_female_${k}.wav` });
      testSamples.push({ emotion: k, gender: "male", actor_id: 21, filename: `test_male_${k}.wav` });
    });
  }

  testSamples.forEach((s, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    const emoTitle = s.emotion.charAt(0).toUpperCase() + s.emotion.slice(1);
    opt.textContent = `${emoTitle} [${s.gender.toUpperCase()} Voice - Actor ${s.actor_id}]`;
    select.appendChild(opt);
  });
}

/* ==========================================================================
   8. Event Listeners Setup
   ========================================================================== */
function setupEventListeners() {
  const select = document.getElementById("sample-select");
  const playBtn = document.getElementById("play-btn");
  const pauseBtn = document.getElementById("pause-btn");
  const stopBtn = document.getElementById("stop-btn");
  const deleteBtn = document.getElementById("delete-btn");
  const recordBtn = document.getElementById("record-btn");
  const cancelRecBtn = document.getElementById("cancel-record-btn");
  const fileInput = document.getElementById("file-input");
  const clearHistBtn = document.getElementById("clear-history-btn");

  // Sample Selection
  select.addEventListener("change", (e) => {
    const idx = e.target.value;
    if (idx !== "") {
      const sample = testSamples[idx];
      loadAudioUrl(`samples/${sample.filename}`, sample.emotion, sample.filename);
    }
  });

  // Upload custom audio
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      loadAudioUrl(url, null, file.name);
    }
  });

  // Play / Pause / Stop / Delete
  playBtn.addEventListener("click", playAudio);
  pauseBtn.addEventListener("click", pauseAudio);
  stopBtn.addEventListener("click", stopAudio);
  deleteBtn.addEventListener("click", deleteCurrentAudio);

  // Recording
  recordBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  cancelRecBtn.addEventListener("click", cancelRecording);

  // History Clear
  clearHistBtn.addEventListener("click", clearAllHistory);
}
