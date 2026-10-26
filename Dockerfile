# Restaurant Analysis – dashboard image (CPU).
#
# Build:  docker build -t restaurant-analysis .
# Run:    docker compose up -d        → http://localhost:8000
#
# Only the browser dashboard runs in the container. The OpenCV window steps
# (phase1/, phase2/, tools/mark_tables.py) need a display and are meant to be
# run directly on the host; inside Docker, tables are drawn on the settings page.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Settings and database live in the /app/data volume so they survive updates.
    RCV_CONFIG=/app/data/config.json \
    RCV_DB=/app/data/restaurant.db \
    # Ultralytics writes its settings here (must be writable by the non-root user).
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HOST=0.0.0.0 \
    PORT=8000

# System libraries OpenCV needs at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first: much smaller than the default CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the YOLO weights at build time so the container starts without internet.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"

# Application code last, so code changes do not invalidate the cached layers above.
COPY . .

# Run as an unprivileged user (uid 1000 matches the first user on most Linux hosts).
RUN useradd --uid 1000 --create-home appuser \
    && mkdir -p /app/data /app/videos \
    && chmod +x docker/entrypoint.sh \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/stats', timeout=4)"

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["python", "main.py"]
