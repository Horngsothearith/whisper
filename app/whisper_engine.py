import os
import sys
import time
import math
import logging
import threading
from typing import Optional, List, Dict, Any, Tuple

# Compatibility patch for PyAV >= 15 where metadata_errors was removed from av.open
try:
    import av
    _orig_av_open = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _patched_av_open
except Exception:
    pass

from faster_whisper import WhisperModel
from app.config import settings, resolve_device_and_compute_type

logger = logging.getLogger("whisper_engine")
logging.basicConfig(level=logging.INFO)

AVAILABLE_MODELS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3"
]

class WhisperEngine:
    """Manages the Faster-Whisper model instance and transcription workflows."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(WhisperEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.model: Optional[WhisperModel] = None
        self.current_model_name: str = settings.WHISPER_MODEL
        self.current_device: str = "cpu"
        self.current_compute_type: str = "int8"
        self.cpu_threads: int = settings.WHISPER_CPU_THREADS
        self.download_root: Optional[str] = settings.WHISPER_DOWNLOAD_ROOT
        self._load_lock = threading.Lock()
        self._initialized = True

    def initialize(self):
        """Loads the initial model configured in settings."""
        device, compute_type = resolve_device_and_compute_type(
            settings.WHISPER_DEVICE, settings.WHISPER_COMPUTE_TYPE
        )
        self.load_model(
            model_name=settings.WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
            cpu_threads=self.cpu_threads
        )

    def load_model(
        self,
        model_name: str,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        cpu_threads: Optional[int] = None
    ) -> Dict[str, Any]:
        """Loads or switches the Whisper model."""
        with self._load_lock:
            # Normalize model alias
            if model_name.lower() == "whisper-1":
                # OpenAI model alias maps to current or turbo/base
                model_name = self.current_model_name or "base"

            if device is None or device == "auto":
                dev, comp = resolve_device_and_compute_type(
                    settings.WHISPER_DEVICE if device is None else device,
                    settings.WHISPER_COMPUTE_TYPE if compute_type is None else compute_type
                )
            else:
                dev = device
                comp = compute_type or ("float16" if dev == "cuda" else "int8")

            threads = cpu_threads or self.cpu_threads
            logger.info(f"Loading Whisper model '{model_name}' on {dev} ({comp}) with {threads} threads...")
            start_t = time.time()
            
            try:
                model_instance = WhisperModel(
                    model_size_or_path=model_name,
                    device=dev,
                    compute_type=comp,
                    cpu_threads=threads,
                    download_root=self.download_root
                )
                self.model = model_instance
                self.current_model_name = model_name
                self.current_device = dev
                self.current_compute_type = comp
                self.cpu_threads = threads
                load_time = time.time() - start_t
                logger.info(f"Model '{model_name}' successfully loaded in {load_time:.2f}s")
                return {
                    "success": True,
                    "model": model_name,
                    "device": dev,
                    "compute_type": comp,
                    "load_time_sec": round(load_time, 2)
                }
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                # If CUDA failed, attempt fallback to CPU
                if dev == "cuda":
                    logger.warning("Attempting CPU fallback...")
                    try:
                        self.model = WhisperModel(
                            model_size_or_path=model_name,
                            device="cpu",
                            compute_type="int8",
                            cpu_threads=threads,
                            download_root=self.download_root
                        )
                        self.current_model_name = model_name
                        self.current_device = "cpu"
                        self.current_compute_type = "int8"
                        return {
                            "success": True,
                            "model": model_name,
                            "device": "cpu",
                            "compute_type": "int8",
                            "note": f"Fell back to CPU because CUDA failed: {str(e)}"
                        }
                    except Exception as fallback_e:
                        raise RuntimeError(f"Failed CPU fallback: {fallback_e}") from e
                raise e

    def transcribe(
        self,
        audio_file_path: str,
        task: str = "transcribe",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: Union[float, List[float]] = 0.0,
        word_timestamps: bool = False,
        vad_filter: bool = True
    ) -> Dict[str, Any]:
        """Runs transcription or translation on the specified audio file."""
        if self.model is None:
            self.initialize()

        # Handle language None or auto
        lang = None if (language in [None, "", "auto"]) else language

        # Format temperature
        if isinstance(temperature, (int, float)):
            temp = [float(temperature)] if float(temperature) != 0.0 else 0.0
        else:
            temp = temperature

        segments_generator, info = self.model.transcribe(
            audio=audio_file_path,
            task=task,
            language=lang,
            initial_prompt=prompt,
            temperature=temp,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500) if vad_filter else None
        )

        segments_list = []
        words_list = []
        full_text_parts = []

        for idx, segment in enumerate(segments_generator):
            seg_words = None
            if word_timestamps and segment.words:
                seg_words = []
                for w in segment.words:
                    word_dict = {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 4) if w.probability is not None else None
                    }
                    seg_words.append(word_dict)
                    words_list.append(word_dict)

            seg_dict = {
                "id": idx,
                "seek": segment.seek,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": segment.text.strip(),
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": round(segment.avg_logprob, 4),
                "compression_ratio": round(segment.compression_ratio, 4),
                "no_speech_prob": round(segment.no_speech_prob, 4),
                "words": seg_words
            }
            segments_list.append(seg_dict)
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts).strip()

        return {
            "task": task,
            "language": info.language,
            "language_probability": round(info.language_probability, 4) if hasattr(info, "language_probability") else 1.0,
            "duration": round(info.duration, 2) if hasattr(info, "duration") else 0.0,
            "text": full_text,
            "words": words_list if word_timestamps else None,
            "segments": segments_list
        }

# Timestamp helpers for SubRip (SRT) and WebVTT (VTT)

def format_timestamp_srt(seconds: float) -> str:
    """Formats seconds into SRT timestamp HH:MM:SS,mmm"""
    millis = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def format_timestamp_vtt(seconds: float) -> str:
    """Formats seconds into WebVTT timestamp HH:MM:SS.mmm"""
    millis = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

def to_srt(segments: List[Dict[str, Any]]) -> str:
    """Converts segments list to standard SubRip (.srt) subtitle string."""
    srt_output = []
    for i, seg in enumerate(segments, start=1):
        start_ts = format_timestamp_srt(seg["start"])
        end_ts = format_timestamp_srt(seg["end"])
        text = seg["text"]
        srt_output.append(f"{i}\n{start_ts} --> {end_ts}\n{text}\n")
    return "\n".join(srt_output)

def to_vtt(segments: List[Dict[str, Any]]) -> str:
    """Converts segments list to standard WebVTT (.vtt) subtitle string."""
    vtt_output = ["WEBVTT\n"]
    for seg in segments:
        start_ts = format_timestamp_vtt(seg["start"])
        end_ts = format_timestamp_vtt(seg["end"])
        text = seg["text"]
        vtt_output.append(f"{start_ts} --> {end_ts}\n{text}\n")
    return "\n".join(vtt_output)

# Global singleton
engine = WhisperEngine()
