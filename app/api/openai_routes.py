import os
import shutil
import tempfile
import logging
from typing import Optional, List, Union
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from app.whisper_engine import engine, to_srt, to_vtt, AVAILABLE_MODELS
from app.api.schemas import (
    TranscriptionJsonResponse,
    VerboseTranscriptionResponse,
    ModelCard,
    ModelListResponse
)

logger = logging.getLogger("openai_routes")
router = APIRouter(prefix="/v1", tags=["OpenAI Compatible Audio API"])

def get_audio_suffix(filename: Optional[str]) -> str:
    """Extracts or defaults audio file extension."""
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[1].lower()
    return ".wav"

@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """Lists models compatible with OpenAI API format."""
    models_data = [
        ModelCard(id="whisper-1", owned_by="openai")
    ]
    for model_name in AVAILABLE_MODELS:
        models_data.append(ModelCard(id=model_name, owned_by="systran"))
    return ModelListResponse(data=models_data)

@router.get("/models/{model_id}", response_model=ModelCard)
async def get_model(model_id: str):
    """Retrieves specific model card."""
    if model_id == "whisper-1" or model_id in AVAILABLE_MODELS:
        return ModelCard(id=model_id, owned_by="systran" if model_id != "whisper-1" else "openai")
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

@router.post("/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(..., description="Audio file to transcribe (wav, mp3, m4a, flac, ogg, webm, etc.)"),
    model: str = Form("whisper-1", description="Model ID (e.g. whisper-1, base, tiny, small, turbo)"),
    language: Optional[str] = Form(None, description="ISO-639-1 language code (e.g. 'en', 'es', 'fr', 'de')"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide transcription style"),
    response_format: str = Form("json", description="json, text, srt, verbose_json, vtt"),
    temperature: float = Form(0.0, description="Sampling temperature between 0 and 1"),
    timestamp_granularities: Optional[List[str]] = Form(None, alias="timestamp_granularities[]", description="Optional: ['segment'], ['word'], or both")
):
    """
    OpenAI-compatible audio transcription endpoint.
    Compatible with openai.audio.transcriptions.create().
    """
    # Check if a specific model was requested and differs from currently loaded
    requested_model = model.strip()
    if requested_model != "whisper-1" and requested_model != engine.current_model_name:
        if requested_model in AVAILABLE_MODELS:
            logger.info(f"Switching active model to requested '{requested_model}'")
            engine.load_model(requested_model)

    # Word timestamps flag
    word_timestamps = False
    if timestamp_granularities and "word" in timestamp_granularities:
        word_timestamps = True
    elif response_format == "verbose_json":
        word_timestamps = True

    # Save uploaded audio to a temporary file
    suffix = get_audio_suffix(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        shutil.copyfileobj(file.file, tmp_file)

    try:
        result = engine.transcribe(
            audio_file_path=tmp_path,
            task="transcribe",
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=True
        )

        fmt = response_format.lower()
        if fmt == "text":
            return PlainTextResponse(result["text"])
        elif fmt == "srt":
            return PlainTextResponse(to_srt(result["segments"]), media_type="text/plain")
        elif fmt == "vtt":
            return PlainTextResponse(to_vtt(result["segments"]), media_type="text/vtt")
        elif fmt == "verbose_json":
            return JSONResponse(result)
        else: # default "json"
            return {"text": result["text"]}
            
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@router.post("/audio/translations")
async def create_translation(
    file: UploadFile = File(..., description="Audio file to translate into English"),
    model: str = Form("whisper-1", description="Model ID (e.g. whisper-1, base, tiny, small, turbo)"),
    prompt: Optional[str] = Form(None, description="Optional prompt to guide translation style"),
    response_format: str = Form("json", description="json, text, srt, verbose_json, vtt"),
    temperature: float = Form(0.0, description="Sampling temperature between 0 and 1")
):
    """
    OpenAI-compatible audio translation endpoint (translates audio to English).
    Compatible with openai.audio.translations.create().
    """
    requested_model = model.strip()
    if requested_model != "whisper-1" and requested_model != engine.current_model_name:
        if requested_model in AVAILABLE_MODELS:
            engine.load_model(requested_model)

    word_timestamps = (response_format.lower() == "verbose_json")
    suffix = get_audio_suffix(file.filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = tmp_file.name
        shutil.copyfileobj(file.file, tmp_file)

    try:
        result = engine.transcribe(
            audio_file_path=tmp_path,
            task="translate",
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
            vad_filter=True
        )

        fmt = response_format.lower()
        if fmt == "text":
            return PlainTextResponse(result["text"])
        elif fmt == "srt":
            return PlainTextResponse(to_srt(result["segments"]), media_type="text/plain")
        elif fmt == "vtt":
            return PlainTextResponse(to_vtt(result["segments"]), media_type="text/vtt")
        elif fmt == "verbose_json":
            return JSONResponse(result)
        else:
            return {"text": result["text"]}
            
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
