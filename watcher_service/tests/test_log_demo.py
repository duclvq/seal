"""
Quick test: embed 1 small video, verify timing logs go to service.log.
"""
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import tempfile
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_SERVICE_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_AUDIOSEAL_SRC = _PROJECT_ROOT / "audioseal" / "src"
if str(_AUDIOSEAL_SRC) not in sys.path:
    sys.path.insert(0, str(_AUDIOSEAL_SRC))

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.chdir(_PROJECT_ROOT)

# ── Setup logging giống service.py (ghi vào service.log) ────────────────────
LOG_DIR = _SERVICE_DIR / "data" / "db"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SERVICE_LOG = LOG_DIR / "service.log"

fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
root = logging.getLogger()
root.setLevel(logging.INFO)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter(fmt))
root.addHandler(sh)

fh = RotatingFileHandler(
    str(SERVICE_LOG), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter(fmt))
root.addHandler(fh)

log = logging.getLogger(__name__)

# ── Run ──────────────────────────────────────────────────────────────────────
import torch
from worker import load_video_model, embed_to_file

CKPT  = "output/run2_video/checkpoint350.pth"
INPUT = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"

device = torch.device("cuda")
log.info("Loading model...")
model = load_video_model(CKPT, device)
log.info("Model loaded.")

OUT_DIR = _SERVICE_DIR / "data" / "test_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = str(OUT_DIR / "DJI_0152_wm.mp4")

log.info(f"Embedding: {INPUT} -> {out_path}")
t0 = time.time()
result = embed_to_file(
    input_path=INPUT,
    output_path=out_path,
    video_model=model,
    watermark_text="test",
    device=device,
    audio_model=None,
)
elapsed = time.time() - t0

log.info(f"Done in {elapsed:.1f}s — {result}")
log.info(f"Output size: {os.path.getsize(out_path)/1024/1024:.1f}MB")
log.info(f"Output saved: {out_path}")

torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"Log file: {SERVICE_LOG}")
print(f"Log size: {SERVICE_LOG.stat().st_size/1024:.1f}KB")
print(f"{'='*60}")
