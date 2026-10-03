# Local Whisper API Server & Transcription Studio

A high-performance, private, 100% offline speech-to-text API server powered by **[Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)** (CTranslate2) and **FastAPI**.

It features full **OpenAI Audio API compatibility** (`/v1/audio/transcriptions` and `/v1/audio/translations`) and an interactive **Web UI Studio** with live microphone recording, audio waveform visualizer, timestamped segment seeking, subtitle exports (SRT, VTT, JSON), and dynamic model switching.

---

## 🌟 Key Features

- **⚡ Blazing Fast Local Inference**: Powered by CTranslate2 with int8 quantization and AVX2 CPU acceleration (up to 4x faster than vanilla Whisper with 50% less RAM).
- **🔌 100% OpenAI Drop-in Compatible**: Use the official `openai` SDK or existing tools (Open WebUI, LibreChat, Cursor, LangChain) simply by setting `base_url="http://localhost:8000/v1"`.
- **🎙️ Interactive Web UI Studio**:
  - In-browser microphone recorder with real-time frequency waveform visualizer.
  - Drag-and-drop audio & video file support (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mp4`).
  - Interactive transcript segments: click any timestamped segment to jump audio playback directly to that position.
  - One-click export to **Plain Text (.txt)**, **SubRip (.srt)**, **WebVTT (.vtt)**, and **Verbose JSON (.json)**.
  - Live hardware resource monitor (CPU %, RAM usage, loaded model, device).
  - Code snippet generator for Python (OpenAI SDK & Requests), cURL, and JavaScript.
- **🛡️ Silero VAD (Voice Activity Detection)**: Built-in voice activity detection to filter background silence and eliminate hallucinated text during pauses.
- **🔄 Dynamic Model Switching**: Switch models on-the-fly (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`) via the Web UI or API without server restarts.
- **📖 Interactive API Documentation**: Swagger UI at `/docs` and ReDoc at `/redoc`.

---

## 🚀 Quick Start

### 1. Run the Server

```bash
# Using the pre-configured virtual environment
.venv\Scripts\python.exe run.py
```

Or customize startup parameters:
```bash
# Run with small model on custom port
.venv\Scripts\python.exe run.py --model small --port 8000

# Force CPU mode with int8 precision
.venv\Scripts\python.exe run.py --model base --device cpu --compute-type int8
```

### 2. Run with Docker Compose

```bash
# Build and start container in detached mode
docker compose up -d

# View live container logs
docker compose logs -f

# Stop container
docker compose down
```

> **Note**: Models are saved to the persistent `whisper-models-cache` Docker volume, so they persist across container rebuilds without re-downloading.

Server URLs:
- **Web UI Studio**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **OpenAI Transcription Endpoint**: `http://localhost:8000/v1/audio/transcriptions`

---

## 💻 API Usage Examples

### 1. Using Official OpenAI Python SDK

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-not-needed"
)

with open("speech.mp3", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-1",  # or "base", "small", "turbo"
        file=audio_file,
        response_format="verbose_json"
    )

print(transcription.text)
```

### 2. Using cURL (SubRip Subtitle Generation)

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@meeting.mp3" \
  -F "model=base" \
  -F "response_format=srt"
```

### 3. Using Python `requests`

```python
import requests

url = "http://localhost:8000/v1/audio/transcriptions"
files = {"file": open("audio.wav", "rb")}
data = {
    "model": "base",
    "language": "en",
    "response_format": "json"
}

res = requests.post(url, files=files, data=data)
print(res.json()["text"])
```

### 4. Audio Translation to English

```bash
curl -X POST "http://localhost:8000/v1/audio/translations" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@french_audio.mp3" \
  -F "model=base" \
  -F "response_format=text"
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible audio transcription (supports `json`, `text`, `srt`, `vtt`, `verbose_json`) |
| `POST` | `/v1/audio/translations` | OpenAI-compatible translation to English |
| `GET` | `/v1/models` | List available models in OpenAI format |
| `GET` | `/api/status` | Server status, hardware utilization (CPU %, RAM), and loaded model |
| `GET` | `/api/models` | List all supported Whisper models |
| `POST` | `/api/models/load` | Switch the active model on the fly (`{"model": "small"}`) |
| `POST` | `/api/transcribe` | Extended transcription endpoint with VAD and timestamp flags |
| `GET` | `/health` | Healthcheck endpoint |

---

## ⚙️ Configuration (`.env`)

You can copy `.env.example` to `.env` to configure default settings:

```env
# Default model: tiny, base, small, medium, large-v3, large-v3-turbo
WHISPER_MODEL=base

# Device: auto, cpu, cuda
WHISPER_DEVICE=auto

# Precision: auto, int8, float16, float32
WHISPER_COMPUTE_TYPE=auto

# CPU threads for CTranslate2
WHISPER_CPU_THREADS=4

# Server host and port
HOST=0.0.0.0
PORT=8000

# Optional API key for bearer auth (leave blank to disable)
API_KEY=
```

---

## 🧪 Testing

Run the automated test suite:
```bash
.venv\Scripts\python.exe tests\test_api.py
```
