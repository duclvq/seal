"""
Verify watermark survives the YUV-native pipeline.
Embed → Extract → Check if watermark is detected and text matches.
"""
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_SERVICE_DIR), str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.chdir(_PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s")
log = logging.getLogger(__name__)

import torch
from worker import load_video_model, embed_to_file
from videoseal.utils.cfg import setup_model_from_checkpoint
from web_demo.core.video_io import load_video_tensor
from web_demo.core.ecc import msg_tensor_to_text_bch

CKPT = "output/run2_video/checkpoint350.pth"
INPUT = r"D:\samu_videos_gpu\DJI_20260104092139_0153_D_gpu_hevc.mp4"
WM_TEXT = "test"

device = torch.device("cuda")

# 1. Load model
log.info("Loading model...")
model = load_video_model(CKPT, device)
log.info("Model loaded.")

# 2. Embed watermark
out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
os.close(out_fd)

log.info(f"Embedding: {INPUT}")
t0 = time.time()
result = embed_to_file(
    input_path=INPUT, output_path=out_path,
    video_model=model, watermark_text=WM_TEXT,
    device=device, audio_model=None,
)
t_embed = time.time() - t0
log.info(f"Embed done in {t_embed:.1f}s — {result}")

# 3. Extract watermark from output
log.info(f"Extracting watermark from: {out_path}")
t0 = time.time()
video, fps = load_video_tensor(out_path, max_frames=900)
n_frames = video.shape[0]
log.info(f"Loaded {n_frames} frames for extraction")

with torch.no_grad():
    msg = model.extract_message(video.to(device), aggregation="squared_avg")

decode = msg_tensor_to_text_bch(msg.cpu())
t_extract = time.time() - t0

# 4. Report
print(f"\n{'='*60}")
print(f"WATERMARK VERIFICATION")
print(f"{'='*60}")
print(f"Input:          {Path(INPUT).name}")
print(f"Embed text:     '{WM_TEXT}'")
print(f"Detected:       {decode['correctable']}")
print(f"Extracted text: '{decode['decoded_text']}'")
print(f"Match:          {'YES' if decode['decoded_text'] == WM_TEXT else 'NO'}")
print(f"Bit accuracy:   {decode.get('bit_accuracy', 'N/A')}")
print(f"Frames used:    {n_frames}")
print(f"Embed time:     {t_embed:.1f}s")
print(f"Extract time:   {t_extract:.1f}s")
print(f"{'='*60}")

# Cleanup
os.unlink(out_path)
torch.cuda.empty_cache()
