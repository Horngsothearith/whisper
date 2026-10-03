from typing import Optional, List, Union, Literal
from pydantic import BaseModel, Field

# OpenAI API Compatible Models

class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float
    probability: Optional[float] = None

class TranscriptionSegment(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0
    words: Optional[List[TranscriptionWord]] = None

class TranscriptionJsonResponse(BaseModel):
    text: str

class VerboseTranscriptionResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: Optional[List[TranscriptionWord]] = None
    segments: List[TranscriptionSegment] = Field(default_factory=list)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 1677610602
    owned_by: str = "local-whisper"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelCard]

# System Management Models

class SystemStatusResponse(BaseModel):
    status: str
    current_model: str
    device: str
    compute_type: str
    cpu_threads: int
    cpu_usage_percent: float
    ram_usage_percent: float
    ram_used_gb: float
    ram_total_gb: float
    cuda_available: bool
    cuda_devices: int
    models_available: List[str]

class ModelLoadRequest(BaseModel):
    model: str
    device: Optional[str] = None
    compute_type: Optional[str] = None

class ModelLoadResponse(BaseModel):
    success: bool
    message: str
    current_model: str
    device: str
    compute_type: str
