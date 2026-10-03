import os
import sys
import argparse
import uvicorn
from app.config import settings

def print_banner(host: str, port: int, model: str, device: str, compute_type: str):
    banner = f"""
====================================================================
                LOCAL WHISPER API SERVER
====================================================================
  * Engine:          Faster-Whisper (CTranslate2 + Silero VAD)
  * Model:           {model}
  * Device:          {device} ({compute_type})
  * Server URL:      http://{host if host != '0.0.0.0' else 'localhost'}:{port}
  * Web Playground:  http://{host if host != '0.0.0.0' else 'localhost'}:{port}/
  * Swagger Docs:    http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs
  * OpenAI Endpoint: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/v1/audio/transcriptions
====================================================================
"""
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="Run the Local Whisper API Server")
    parser.add_argument("--host", default=settings.HOST, help="Host to bind server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=settings.PORT, help="Port to bind server (default: 8000)")
    parser.add_argument("--model", default=settings.WHISPER_MODEL, help="Whisper model to load (e.g. tiny, base, small, turbo)")
    parser.add_argument("--device", default=settings.WHISPER_DEVICE, help="Device to use: auto, cpu, cuda")
    parser.add_argument("--compute-type", default=settings.WHISPER_COMPUTE_TYPE, help="Compute precision: auto, int8, float16, float32")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    # Override environment settings with CLI flags
    os.environ["WHISPER_MODEL"] = args.model
    os.environ["WHISPER_DEVICE"] = args.device
    os.environ["WHISPER_COMPUTE_TYPE"] = args.compute_type
    settings.WHISPER_MODEL = args.model
    settings.WHISPER_DEVICE = args.device
    settings.WHISPER_COMPUTE_TYPE = args.compute_type

    print_banner(args.host, args.port, args.model, args.device, args.compute_type)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
