import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from app.config import settings
from app.whisper_engine import engine
from app.api.openai_routes import router as openai_router
from app.api.system_routes import router as system_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("whisper_app")

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if settings.API_KEY:
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key.")
        clean_key = api_key.replace("Bearer ", "").strip()
        if clean_key != settings.API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key.")
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the whisper engine in background or eagerly
    logger.info("Initializing Local Whisper Engine on startup...")
    try:
        engine.initialize()
        logger.info(f"Whisper engine ready (model: {engine.current_model_name}, device: {engine.current_device}, compute_type: {engine.current_compute_type})")
    except Exception as e:
        logger.error(f"Error during initial model load: {e}")
    yield
    # Shutdown
    logger.info("Local Whisper API shutting down.")

app = FastAPI(
    title="Local Whisper API Server",
    description="High-performance, local speech-to-text API powered by Faster-Whisper. Features full OpenAI API compatibility and an interactive audio playground.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for cross-origin web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(openai_router, dependencies=[Depends(verify_api_key)])
app.include_router(system_router, dependencies=[Depends(verify_api_key)])

# Static directory setup
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
async def serve_playground():
    """Serves the interactive Web UI playground."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({
        "status": "online",
        "message": "Local Whisper API Server is running. Visit /docs for Swagger API documentation."
    })

@app.get("/health", tags=["Health"])
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "model": engine.current_model_name,
        "device": engine.current_device,
        "compute_type": engine.current_compute_type
    }
