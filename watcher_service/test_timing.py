"""
Quick timing test — runs embed_to_file on a single video.
Usage:  python watcher_service/test_timing.py
"""
import os, sys, time, logging
from pathlib import Path

# ── Setup paths ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "audioseal" / "src"))
os.environ.setdefault("NO_TORCH_COMPILE", "1")

# ── Load .env_home for local config ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    env_home = _HERE / ".env_home"
    if env_home.exists():
        load_dotenv(env_home, override=True)
except ImportError:
    pass

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

import torch

# ── Pick a test video ────────────────────────────────────────────────────────
TEST_VIDEO = _ROOT / "data" / "val" / "gi-diploma-66013582898105388892353.mp4"
OUTPUT_DIR = _ROOT / "output" / "timing_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / TEST_VIDEO.name

if not TEST_VIDEO.exists():
    log.error(f"Test video not found: {TEST_VIDEO}")
    sys.exit(1)

# ── Load models ──────────────────────────────────────────────────────────────
from worker import load_video_model, load_audio_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = os.getenv("CUSTOM_CKPT_PATH", "output/run2_video/checkpoint350.pth")
if not Path(ckpt).is_absolute():
    ckpt = str(_ROOT / ckpt)

log.info(f"Device: {device}")
log.info(f"Loading VideoSeal from {ckpt} ...")
video_model = load_video_model(ckpt, device)
log.info("VideoSeal ready.")

log.info("Loading AudioSeal ...")
audio_model = load_audio_model(device)
log.info(f"AudioSeal: {'ready' if audio_model else 'unavailable'}")

# ── Warmup (first run is always slower due to CUDA lazy init) ────────────────
log.info("Warmup run ...")
from worker import embed_to_file
_ = embed_to_file(
    input_path=str(TEST_VIDEO),
    output_path=str(OUTPUT_DIR / "warmup.mp4"),
    video_model=video_model,
    watermark_text="warm",
    device=device,
    audio_model=audio_model,
)
# Clean up warmup output
warmup_f = OUTPUT_DIR / "warmup.mp4"
if warmup_f.exists():
    warmup_f.unlink()
log.info("Warmup done.\n")

# ── Actual timed run ─────────────────────────────────────────────────────────
log.info(f"=" * 70)
log.info(f"TIMING RUN: {TEST_VIDEO.name}")
log.info(f"=" * 70)

t_total_start = time.time()
result = embed_to_file(
    input_path=str(TEST_VIDEO),
    output_path=str(OUTPUT_PATH),
    video_model=video_model,
    watermark_text="vtvwm0",
    device=device,
    audio_model=audio_model,
)
t_total = time.time() - t_total_start

log.info(f"=" * 70)
log.info(f"RESULT: {result}")
log.info(f"TOTAL WALL TIME: {t_total:.3f}s")
log.info(f"=" * 70)
