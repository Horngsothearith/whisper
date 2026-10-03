import os
import sys
from fastapi.testclient import TestClient

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.whisper_engine import engine
from tests.generate_test_audio import create_synthetic_wav

client = TestClient(app)

def setup_module():
    """Ensure engine is loaded with lightweight model for testing."""
    os.environ["WHISPER_MODEL"] = "tiny"
    try:
        engine.load_model("tiny", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"Engine preload warning: {e}")

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model" in data

def test_system_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ["ready", "initializing"]
    assert "cpu_usage_percent" in data
    assert "ram_used_gb" in data
    assert len(data["models_available"]) > 0

def test_models_list():
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "whisper-1" in ids
    assert "tiny" in ids

def test_transcription_endpoints():
    test_wav = os.path.join(os.path.dirname(__file__), "sample.wav")
    create_synthetic_wav(test_wav, duration_sec=1.0)

    try:
        # 1. JSON format (OpenAI default)
        with open(test_wav, "rb") as f:
            res = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", f, "audio/wav")},
                data={"model": "tiny", "response_format": "json"}
            )
        assert res.status_code == 200
        assert "text" in res.json()

        # 2. Text format
        with open(test_wav, "rb") as f:
            res = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", f, "audio/wav")},
                data={"model": "tiny", "response_format": "text"}
            )
        assert res.status_code == 200
        assert isinstance(res.text, str)

        # 3. Verbose JSON format
        with open(test_wav, "rb") as f:
            res = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", f, "audio/wav")},
                data={"model": "tiny", "response_format": "verbose_json"}
            )
        assert res.status_code == 200
        v_data = res.json()
        assert "task" in v_data
        assert "language" in v_data
        assert "segments" in v_data

        # 4. SRT format
        with open(test_wav, "rb") as f:
            res = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", f, "audio/wav")},
                data={"model": "tiny", "response_format": "srt"}
            )
        assert res.status_code == 200

        # 5. Translation endpoint
        with open(test_wav, "rb") as f:
            res = client.post(
                "/v1/audio/translations",
                files={"file": ("sample.wav", f, "audio/wav")},
                data={"model": "tiny", "response_format": "json"}
            )
        assert res.status_code == 200
        assert "text" in res.json()

    finally:
        if os.path.exists(test_wav):
            try:
                os.remove(test_wav)
            except Exception:
                pass

if __name__ == "__main__":
    setup_module()
    print("Running test_health...")
    test_health()
    print("Running test_system_status...")
    test_system_status()
    print("Running test_models_list...")
    test_models_list()
    print("Running test_transcription_endpoints...")
    test_transcription_endpoints()
    print("ALL TESTS PASSED SUCCESSFULLY!")
