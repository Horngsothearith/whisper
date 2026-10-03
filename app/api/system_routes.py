import os
import shutil
import tempfile
import psutil
import ctranslate2
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.whisper_engine import engine, AVAILABLE_MODELS
from app.api.schemas import (
    SystemStatusResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    VerboseTranscriptionResponse
)

router = APIRouter(prefix="/api", tags=["System & Engine Management"])

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Returns current server status, hardware utilization, and active model info."""
    mem = psutil.virtual_memory()
    cuda_devices = 0
    try:
        cuda_devices = ctranslate2.get_cuda_device_count()
    except Exception:
        pass

    return SystemStatusResponse(
        status="ready" if engine.model is not None else "initializing",
        current_model=engine.current_model_name,
        device=engine.current_device,
        compute_type=engine.current_compute_type,
        cpu_threads=engine.cpu_threads,
        cpu_usage_percent=psutil.cpu_percent(interval=None),
        ram_usage_percent=mem.percent,
        ram_used_gb=round(mem.used / (1024 ** 3), 2),
        ram_total_gb=round(mem.total / (1024 ** 3), 2),
        cuda_available=(cuda_devices > 0),
        cuda_devices=cuda_devices,
        models_available=AVAILABLE_MODELS
    )

@router.get("/models")
async def get_available_models():
    """Returns list of all supported Whisper models and current active model."""
    return {
        "current_model": engine.current_model_name,
        "available_models": AVAILABLE_MODELS
    }

@router.post("/models/load", response_model=ModelLoadResponse)
async def switch_model(req: ModelLoadRequest):
    """Dynamically loads or switches the active Whisper model."""
    model_name = req.model.strip()
    if model_name not in AVAILABLE_MODELS and model_name != "whisper-1":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not in the list of supported models: {AVAILABLE_MODELS}"
        )

    try:
        res = engine.load_model(
            model_name=model_name,
            device=req.device,
            compute_type=req.compute_type
        )
        return ModelLoadResponse(
            success=True,
            message=f"Model '{model_name}' successfully loaded.",
            current_model=engine.current_model_name,
            device=engine.current_device,
            compute_type=engine.current_compute_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/transcribe", response_model=VerboseTranscriptionResponse)
async def advanced_transcribe(
    file: UploadFile = File(..., description="Audio file"),
    task: str = Form("transcribe", description="'transcribe' or 'translate'"),
    language: Optional[str] = Form(None, description="Audio language code (e.g. 'en') or null for auto"),
    prompt: Optional[str] = Form(None, description="Initial prompt / context"),
    temperature: float = Form(0.0, description="Temperature"),
    word_timestamps: bool = Form(True, description="Calculate word-level timestamps"),
    vad_filter: bool = Form(True, description="Apply Silero VAD to remove silence/noise")
):
    """
    Direct rich transcription endpoint returning detailed segments, confidence, and word timestamps.
    """
    suffix = ".wav"
    if file.filename and "." in file.filename:
        suffix = "." + file.filename.rsplit(".", 1)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        shutil.copyfileobj(file.file, tmp_file)

    try:
        result = engine.transcribe(
            audio_file_path=tmp_path,
            task=task,
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter
        )
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
