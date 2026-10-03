FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WHISPER_DOWNLOAD_ROOT=/models \
    HOST=0.0.0.0 \
    PORT=8000

# Install ffmpeg (required for audio decoding) and curl (for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app/ app/
COPY run.py .

# Create volume mount point for model caching
RUN mkdir -p /models

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the server
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
