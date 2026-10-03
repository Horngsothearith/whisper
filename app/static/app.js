// State management
let currentAudioBlob = null;
let currentAudioFile = null;
let activeInputTab = 'mic';
let activeOutputTab = 'text';
let lastTranscriptionResult = null;
let lastSrtContent = '';
let lastVttContent = '';

// MediaRecorder & Web Audio API Visualizer State
let mediaRecorder = null;
let recordedChunks = [];
let audioContext = null;
let analyser = null;
let visualizerAnimationId = null;
let recordStartTime = null;
let recordTimerInterval = null;

// DOM Elements
const engineStatusPill = document.getElementById('engine-status-pill');
const engineStatusText = document.getElementById('engine-status-text');
const currentModelTag = document.getElementById('current-model-tag');
const currentDeviceTag = document.getElementById('current-device-tag');
const headerModelSelect = document.getElementById('header-model-select');
const switchModelBtn = document.getElementById('switch-model-btn');
const cpuNum = document.getElementById('cpu-num');
const cpuBar = document.getElementById('cpu-bar');
const ramNum = document.getElementById('ram-num');
const ramBar = document.getElementById('ram-bar');

const tabMic = document.getElementById('tab-mic');
const tabUpload = document.getElementById('tab-upload');
const contentMic = document.getElementById('content-mic');
const contentUpload = document.getElementById('content-upload');

const recordBtn = document.getElementById('record-btn');
const recordBtnText = document.getElementById('record-btn-text');
const resetAudioBtn = document.getElementById('reset-audio-btn');
const recordTimer = document.getElementById('record-timer');
const visualizerCanvas = document.getElementById('visualizer-canvas');
const canvasCtx = visualizerCanvas.getContext('2d');

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const browseBtn = document.getElementById('browse-btn');

const audioPreviewSection = document.getElementById('audio-preview-section');
const audioPreview = document.getElementById('audio-preview');
const previewFileName = document.getElementById('preview-file-name');

const paramLanguage = document.getElementById('param-language');
const paramTask = document.getElementById('param-task');
const paramVad = document.getElementById('param-vad');
const paramTimestamps = document.getElementById('param-timestamps');
const paramPrompt = document.getElementById('param-prompt');
const transcribeBtn = document.getElementById('transcribe-btn');
const transcribeBtnText = document.getElementById('transcribe-btn-text');

const outputMetaStrip = document.getElementById('output-meta-strip');
const metaLang = document.getElementById('meta-lang');
const metaDuration = document.getElementById('meta-duration');
const metaSpeed = document.getElementById('meta-speed');
const copyBtn = document.getElementById('copy-btn');
const downloadBtn = document.getElementById('download-btn');

const emptyState = document.getElementById('empty-state');
const transcriptionText = document.getElementById('transcription-text');
const segmentsList = document.getElementById('segments-list');
const srtOutput = document.getElementById('srt-output');
const vttOutput = document.getElementById('vtt-output');
const jsonOutput = document.getElementById('json-output');

const apiToggleBtn = document.getElementById('api-toggle-btn');
const apiCardHeader = document.getElementById('api-card-header');
const apiCardBody = document.getElementById('api-card-body');
const codeSnippetDisplay = document.getElementById('code-snippet-display');
const toastEl = document.getElementById('toast');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initRecorder();
  initDropzone();
  initOutputTabs();
  initApiSnippets();
  fetchSystemStatus();
  setInterval(fetchSystemStatus, 4000);
  drawIdleVisualizer();
});

// Toast notification helper
function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), 2500);
}

// --- Status & Model Polling ---
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('API unreachable');
    const data = await res.json();

    engineStatusText.textContent = 'Engine Ready';
    engineStatusPill.style.background = 'rgba(16, 185, 129, 0.1)';
    engineStatusPill.style.borderColor = 'rgba(16, 185, 129, 0.25)';

    currentModelTag.textContent = `model: ${data.current_model}`;
    currentDeviceTag.textContent = `${data.device.toUpperCase()} (${data.compute_type})`;
    headerModelSelect.value = data.current_model;

    cpuNum.textContent = `${Math.round(data.cpu_usage_percent)}%`;
    cpuBar.style.width = `${Math.min(100, Math.round(data.cpu_usage_percent))}%`;

    ramNum.textContent = `${data.ram_used_gb} / ${data.ram_total_gb} GB`;
    ramBar.style.width = `${Math.min(100, Math.round(data.ram_usage_percent))}%`;
  } catch (err) {
    engineStatusText.textContent = 'Connecting...';
    engineStatusPill.style.background = 'rgba(239, 68, 68, 0.1)';
    engineStatusPill.style.borderColor = 'rgba(239, 68, 68, 0.3)';
  }
}

// Model switcher button
switchModelBtn.addEventListener('click', async () => {
  const chosenModel = headerModelSelect.value;
  switchModelBtn.disabled = true;
  switchModelBtn.textContent = 'Loading...';
  showToast(`Loading Whisper '${chosenModel}' model...`);

  try {
    const res = await fetch('/api/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: chosenModel })
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`Model '${data.current_model}' loaded on ${data.device}!`);
      fetchSystemStatus();
      updateCodeSnippet();
    } else {
      showToast(`Error: ${data.detail || 'Could not load model'}`);
    }
  } catch (e) {
    showToast(`Network error: ${e.message}`);
  } finally {
    switchModelBtn.disabled = false;
    switchModelBtn.textContent = 'Switch';
  }
});

// --- Tab Switching ---
function initTabs() {
  tabMic.addEventListener('click', () => switchInputTab('mic'));
  tabUpload.addEventListener('click', () => switchInputTab('upload'));
}

function switchInputTab(tab) {
  activeInputTab = tab;
  if (tab === 'mic') {
    tabMic.classList.add('active');
    tabUpload.classList.remove('active');
    contentMic.classList.add('active');
    contentUpload.classList.remove('active');
  } else {
    tabUpload.classList.add('active');
    tabMic.classList.remove('active');
    contentUpload.classList.add('active');
    contentMic.classList.remove('active');
  }
}

// --- Audio Visualizer & Recording ---
function initRecorder() {
  recordBtn.addEventListener('click', toggleRecording);
  resetAudioBtn.addEventListener('click', resetAudio);
}

function drawIdleVisualizer() {
  const w = visualizerCanvas.width;
  const h = visualizerCanvas.height;
  canvasCtx.clearRect(0, 0, w, h);
  
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = 'rgba(99, 102, 241, 0.25)';
  canvasCtx.beginPath();
  const sliceWidth = w / 40;
  let x = 0;
  for (let i = 0; i < 40; i++) {
    const y = h / 2 + Math.sin(i * 0.5) * 4;
    if (i === 0) canvasCtx.moveTo(x, y);
    else canvasCtx.lineTo(x, y);
    x += sliceWidth;
  }
  canvasCtx.stroke();
}

async function toggleRecording() {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    // Stop recording
    mediaRecorder.stop();
    recordBtn.classList.remove('recording');
    recordBtnText.textContent = 'Record Again';
    clearInterval(recordTimerInterval);
    if (visualizerAnimationId) cancelAnimationFrame(visualizerAnimationId);
    drawIdleVisualizer();
  } else {
    // Start recording
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      recordedChunks = [];
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/ogg';

      mediaRecorder = new MediaRecorder(stream, { mimeType });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) recordedChunks.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: mimeType });
        currentAudioBlob = blob;
        currentAudioFile = new File([blob], 'microphone_recording.webm', { type: mimeType });
        setAudioPreview(currentAudioBlob, 'Microphone Audio (' + (recordedChunks.length) + ' chunks)');
        // Stop audio tracks
        stream.getTracks().forEach(track => track.stop());
        if (audioContext && audioContext.state !== 'closed') {
          audioContext.close();
        }
      };

      mediaRecorder.start(200);
      recordBtn.classList.add('recording');
      recordBtnText.textContent = 'Stop Recording';
      recordStartTime = Date.now();
      resetAudioBtn.style.display = 'inline-block';

      recordTimerInterval = setInterval(updateRecordTimer, 100);
      visualize();
    } catch (err) {
      alert('Microphone access denied or not available: ' + err.message);
    }
  }
}

function updateRecordTimer() {
  const elapsed = (Date.now() - recordStartTime) / 1000;
  const mins = Math.floor(elapsed / 60);
  const secs = (elapsed % 60).toFixed(1);
  recordTimer.textContent = `${String(mins).padStart(2, '0')}:${secs < 10 ? '0' : ''}${secs}`;
}

function visualize() {
  if (!analyser) return;
  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    visualizerAnimationId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    const w = visualizerCanvas.width;
    const h = visualizerCanvas.height;
    canvasCtx.clearRect(0, 0, w, h);

    const barWidth = (w / bufferLength) * 2.2;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * h * 0.9;
      const grad = canvasCtx.createLinearGradient(0, h, 0, 0);
      grad.addColorStop(0, '#6366f1');
      grad.addColorStop(1, '#ec4899');
      canvasCtx.fillStyle = grad;
      canvasCtx.fillRect(x, h - barHeight, barWidth, barHeight);
      x += barWidth + 2;
    }
  }
  draw();
}

// --- Dropzone & File Handling ---
function initDropzone() {
  dropzone.addEventListener('click', () => fileInput.click());
  browseBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });
}

function handleSelectedFile(file) {
  currentAudioFile = file;
  currentAudioBlob = file;
  setAudioPreview(file, file.name);
}

function setAudioPreview(blobOrFile, name) {
  const url = URL.createObjectURL(blobOrFile);
  audioPreview.src = url;
  previewFileName.textContent = name;
  audioPreviewSection.style.display = 'flex';
  transcribeBtn.disabled = false;
  resetAudioBtn.style.display = 'inline-block';
}

function resetAudio() {
  currentAudioFile = null;
  currentAudioBlob = null;
  audioPreview.src = '';
  audioPreviewSection.style.display = 'none';
  transcribeBtn.disabled = true;
  resetAudioBtn.style.display = 'none';
  recordTimer.textContent = '00:00.0';
  recordBtnText.textContent = 'Start Recording';
  recordBtn.classList.remove('recording');
  fileInput.value = '';
  drawIdleVisualizer();
}

// --- Output View Handling ---
function initOutputTabs() {
  const tabs = ['text', 'segments', 'srt', 'vtt', 'json'];
  tabs.forEach(tab => {
    const btn = document.getElementById(`out-tab-${tab}`);
    btn.addEventListener('click', () => {
      tabs.forEach(t => {
        document.getElementById(`out-tab-${t}`).classList.remove('active');
        document.getElementById(`view-${t}`).classList.remove('active');
      });
      btn.classList.add('active');
      document.getElementById(`view-${tab}`).classList.add('active');
      activeOutputTab = tab;
    });
  });

  copyBtn.addEventListener('click', handleCopy);
  downloadBtn.addEventListener('click', handleDownload);
}

// --- Transcribe Action ---
transcribeBtn.addEventListener('click', runTranscription);

async function runTranscription() {
  if (!currentAudioFile) {
    showToast('Please record audio or select a file first.');
    return;
  }

  transcribeBtn.disabled = true;
  transcribeBtnText.textContent = 'Transcribing...';
  const startTime = Date.now();

  const formData = new FormData();
  formData.append('file', currentAudioFile, currentAudioFile.name || 'audio.wav');
  formData.append('task', paramTask.value);
  if (paramLanguage.value !== 'auto') {
    formData.append('language', paramLanguage.value);
  }
  if (paramPrompt.value.trim()) {
    formData.append('prompt', paramPrompt.value.trim());
  }
  formData.append('vad_filter', paramVad.checked ? 'true' : 'false');
  formData.append('word_timestamps', paramTimestamps.checked ? 'true' : 'false');
  formData.append('temperature', '0.0');

  try {
    const res = await fetch('/api/transcribe', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Transcription failed');
    }

    const data = await res.json();
    const elapsedSec = ((Date.now() - startTime) / 1000).toFixed(2);
    lastTranscriptionResult = data;

    renderResults(data, elapsedSec);
    showToast(`Transcribed in ${elapsedSec}s!`);
  } catch (err) {
    alert('Transcription error: ' + err.message);
  } finally {
    transcribeBtn.disabled = false;
    transcribeBtnText.textContent = 'Transcribe Audio';
  }
}

function renderResults(data, elapsedSec) {
  emptyState.style.display = 'none';
  transcriptionText.style.display = 'block';
  outputMetaStrip.style.display = 'flex';

  metaLang.textContent = (data.language || 'unknown').toUpperCase();
  metaDuration.textContent = `${data.duration || 0}s`;
  metaSpeed.textContent = `${elapsedSec}s`;

  // Text View
  transcriptionText.textContent = data.text || '(No speech detected)';

  // Segments View
  segmentsList.innerHTML = '';
  if (data.segments && data.segments.length > 0) {
    data.segments.forEach(seg => {
      const div = document.createElement('div');
      div.className = 'segment-item';
      div.innerHTML = `
        <div class="segment-meta">
          <span class="timestamp-badge">${formatTime(seg.start)} ➔ ${formatTime(seg.end)}</span>
        </div>
        <div class="segment-text">${escapeHtml(seg.text)}</div>
      `;
      // Click to jump audio
      div.addEventListener('click', () => {
        if (audioPreview && audioPreview.src) {
          audioPreview.currentTime = seg.start;
          audioPreview.play();
        }
      });
      segmentsList.appendChild(div);
    });
  } else {
    segmentsList.innerHTML = '<p class="hint-text">No segments generated.</p>';
  }

  // Generate SRT & VTT
  lastSrtContent = generateSrt(data.segments || []);
  lastVttContent = generateVtt(data.segments || []);
  srtOutput.textContent = lastSrtContent;
  vttOutput.textContent = lastVttContent;

  // JSON View
  jsonOutput.textContent = JSON.stringify(data, null, 2);
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(1);
  return `${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`;
}

function generateSrt(segments) {
  return segments.map((seg, i) => {
    const start = formatSrtTimestamp(seg.start);
    const end = formatSrtTimestamp(seg.end);
    return `${i + 1}\n${start} --> ${end}\n${seg.text}\n`;
  }).join('\n');
}

function generateVtt(segments) {
  const lines = ['WEBVTT\n'];
  segments.forEach(seg => {
    const start = formatVttTimestamp(seg.start);
    const end = formatVttTimestamp(seg.end);
    lines.push(`${start} --> ${end}\n${seg.text}\n`);
  });
  return lines.join('\n');
}

function formatSrtTimestamp(sec) {
  const millis = Math.round((sec - Math.floor(sec)) * 1000);
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const remSec = s % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(remSec).padStart(2,'0')},${String(millis).padStart(3,'0')}`;
}

function formatVttTimestamp(sec) {
  const millis = Math.round((sec - Math.floor(sec)) * 1000);
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const remSec = s % 60;
  return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(remSec).padStart(2,'0')}.${String(millis).padStart(3,'0')}`;
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Copy & Download
function handleCopy() {
  let content = '';
  if (activeOutputTab === 'text') content = transcriptionText.textContent;
  else if (activeOutputTab === 'segments') content = transcriptionText.textContent;
  else if (activeOutputTab === 'srt') content = lastSrtContent;
  else if (activeOutputTab === 'vtt') content = lastVttContent;
  else if (activeOutputTab === 'json') content = jsonOutput.textContent;

  if (!content) {
    showToast('Nothing to copy yet.');
    return;
  }
  navigator.clipboard.writeText(content).then(() => {
    showToast('Copied to clipboard!');
  });
}

function handleDownload() {
  let content = '';
  let filename = 'transcription';
  let mimeType = 'text/plain';

  if (activeOutputTab === 'text' || activeOutputTab === 'segments') {
    content = transcriptionText.textContent;
    filename += '.txt';
  } else if (activeOutputTab === 'srt') {
    content = lastSrtContent;
    filename += '.srt';
  } else if (activeOutputTab === 'vtt') {
    content = lastVttContent;
    filename += '.vtt';
  } else if (activeOutputTab === 'json') {
    content = jsonOutput.textContent;
    filename += '.json';
    mimeType = 'application/json';
  }

  if (!content) {
    showToast('No content to download.');
    return;
  }

  const blob = new Blob([content], { type: mimeType });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  showToast(`Downloaded ${filename}`);
}

// --- API Snippets Accordion & Code Generator ---
let activeSnippetLang = 'python-openai';

function initApiSnippets() {
  apiCardHeader.addEventListener('click', () => {
    const isHidden = apiCardBody.style.display === 'none';
    apiCardBody.style.display = isHidden ? 'block' : 'none';
    apiToggleBtn.textContent = isHidden ? '▲' : '▼';
  });

  const subtabs = document.querySelectorAll('.api-subtab');
  subtabs.forEach(btn => {
    btn.addEventListener('click', () => {
      subtabs.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeSnippetLang = btn.dataset.lang;
      updateCodeSnippet();
    });
  });

  updateCodeSnippet();
}

function updateCodeSnippet() {
  const origin = window.location.origin || 'http://localhost:8000';
  const model = headerModelSelect.value || 'base';

  let code = '';
  if (activeSnippetLang === 'python-openai') {
    code = `# Use the official OpenAI SDK pointing to your local Whisper server!
from openai import OpenAI

client = OpenAI(
    base_url="${origin}/v1",
    api_key="local-not-needed"  # Local server does not require key by default
)

with open("speech.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="${model}",       # or "whisper-1"
        file=audio_file,
        response_format="verbose_json"
    )

print(transcription.text)`;
  } else if (activeSnippetLang === 'python-requests') {
    code = `import requests

url = "${origin}/v1/audio/transcriptions"
files = {"file": open("speech.mp3", "rb")}
data = {"model": "${model}", "response_format": "json"}

response = requests.post(url, files=files, data=data)
print(response.json()["text"])`;
  } else if (activeSnippetLang === 'curl') {
    code = `# Direct cURL call (OpenAI standard audio transcription)
curl -X POST "${origin}/v1/audio/transcriptions" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@speech.mp3" \\
  -F "model=${model}" \\
  -F "response_format=json"`;
  } else if (activeSnippetLang === 'js') {
    code = `const formData = new FormData();
formData.append("file", audioBlobOrFile);
formData.append("model", "${model}");

const response = await fetch("${origin}/v1/audio/transcriptions", {
  method: "POST",
  body: formData
});

const result = await response.json();
console.log(result.text);`;
  }

  codeSnippetDisplay.textContent = code;
}
