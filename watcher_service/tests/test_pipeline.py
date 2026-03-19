"""
Comprehensive test suite for the VideoSeal watcher pipeline.
Tests: watermark, quality, stability, flicker, frame count, file size,
       params, format support (MXF/WebM), and output extension preservation.

Usage:
    python watcher_service/tests/test_pipeline.py
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_TESTS_DIR    = Path(__file__).resolve().parent
_SERVICE_DIR  = _TESTS_DIR.parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_SERVICE_DIR), str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.chdir(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

import av
import numpy as np
import torch

from worker import load_video_model, load_audio_model, embed_to_file
from web_demo.core.ecc import msg_tensor_to_text_bch

# ── Config ───────────────────────────────────────────────────────────────────
CKPT        = "output/run2_video/checkpoint350.pth"
INPUT_VIDEO = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
FIXTURES    = _TESTS_DIR / "fixtures"
WM_TEXT     = "test"
DEVICE      = torch.device("cuda")

# Thresholds
MIN_PSNR_DB        = 35.0
MAX_FILESIZE_DIFF  = 0.05
MAX_FRAME_DROP     = 2
FLICKER_THRESHOLD  = 3.0
EXTRACT_FRAMES     = 300
EXTRACT_CHUNK      = 8

# ── Globals ──────────────────────────────────────────────────────────────────
_model = None
_audio_model = None
_output_path = None          # cached mp4 output for tests 1-8
_results = {}
_cleanup = []                # temp files to clean up at end
