# ── VideoSeal Watcher Service ─────────────────────────────────────────────────
# Matches local env: CUDA 12.8 · PyTorch 2.7.1+cu128 · Python 3.12
#
# Build from the PROJECT ROOT (not inside watcher_service/):
#   docker compose -f watcher_service/docker-compose.yml up --build
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        ffmpeg \
        git \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 / python point to the system Python (3.10 on Ubuntu 22.04)
RUN ln -sf /usr/bin/python3 /usr/bin/python \
 && pip3 install --no-cache-dir --upgrade pip

# ── PyTorch 2.7.1 + CUDA 12.8 ────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Service requirements ──────────────────────────────────────────────────────
COPY watcher_service/requirements.txt /tmp/req_watcher.txt
RUN pip install --no-cache-dir -r /tmp/req_watcher.txt

# ── Copy project source ───────────────────────────────────────────────────────
WORKDIR /app

COPY videoseal/       /app/videoseal/
COPY web_demo/        /app/web_demo/
COPY audioseal/       /app/audioseal/
COPY configs/         /app/configs/
COPY watcher_service/ /app/watcher_service/
COPY extract_app/     /app/extract_app/
COPY pyproject.toml   /app/pyproject.toml

# Install the videoseal package in editable mode
RUN pip install --no-cache-dir -e /app

# ── Volumes ───────────────────────────────────────────────────────────────────
# /data/input       — watched input folder  (mounted from host)
# /data/output      — watermarked output    (mounted from host)
# /data             — DB + error.log        (mounted from host)
# /app/output       — model checkpoints     (mounted from host, read-only)
VOLUME ["/data", "/app/output"]

# ── Environment defaults (overridden by .env / docker-compose) ────────────────
ENV INPUT_FOLDER=/data/input       \
    OUTPUT_FOLDER=/data/output     \
    DB_PATH=/data/watermarks.db    \
    ERROR_LOG=/data/error.log      \
    DEVICE=cuda                    \
    PYTHONUNBUFFERED=1             \
    NO_TORCH_COMPILE=1

CMD ["python", "watcher_service/service.py"]
