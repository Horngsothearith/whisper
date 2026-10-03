import os
from typing import Optional, List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model configuration
    WHISPER_MODEL: str = "base"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"
    WHISPER_CPU_THREADS: int = 4
    WHISPER_DOWNLOAD_ROOT: Optional[str] = None
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    
    # Optional API key protection
    API_KEY: Optional[str] = None
    
    # VAD & transcription defaults
    ENABLE_VAD: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()

def resolve_device_and_compute_type(device_setting: str, compute_setting: str):
    """Resolves 'auto' into concrete device and compute_type for CTranslate2."""
    import ctranslate2
    
    device = device_setting.lower()
    if device == "auto":
        # Check if CUDA device count is > 0 and CUDA is functional
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
            
    compute_type = compute_setting.lower()
    if compute_type == "auto":
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"
            
    return device, compute_type
