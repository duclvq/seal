"""
Full round-trip test: Embed watermark → Extract (streaming) → Verify detection.
Tests the entire pipeline end-to-end.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "audioseal" / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "watcher_service"))
os.chdir(_PROJECT_ROOT)
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import torch
import av
from videoseal.utils.cfg import setup_model_from_checkpoint
from web_demo.core.ecc import msg_tensor_to_text_bch
from worker import load_video_model, embed_to_file

CKPT = "output/run2_video/checkpoint350.pth"
INPUT = r"D:\samu_videos_gpu\DJI_20260104092139_0153_D_gpu_hevc.mp4"
WM_TEXT = "test"
CHUNK_SIZE = 30
MAX_FRAMES = 900

device = torch.device("cuda")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: EMBED
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: EMBED WATERMARK")
print("=" * 60)

model = load_video_model(CKPT, device)

out_fd, out_path = tempfile.mkstemp(suffix=".mp4")
os.close(out_fd)

print(f"Input:  {INPUT}")
print(f"Output: {out_path}")
print(f"Text:   '{WM_TEXT}'")

t0 = time.time()
result = embed_to_file(
    input_path=INPUT,
    output_path=out_path,
    video_model=model,
    watermark_text=WM_TEXT,
    device=device,
    audio_model=None,
)
t_embed = time.time() - t0

out_size = os.path.getsize(out_path) / 1024 / 1024
print(f"\nEmbed done: {t_embed:.1f}s")
print(f"  Mode:       {result['mode']}")
print(f"  Frames:     {result['total_frames']}")
print(f"  Resolution: {result['resolution']}")
print(f"  FPS:        {result['fps']}")
print(f"  Speed:      {result['total_frames']/t_embed:.0f} f/s")
print(f"  Output:     {out_size:.1f}MB")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: EXTRACT (streaming chunk-by-chunk)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("STEP 2: EXTRACT WATERMARK (streaming)")
print("=" * 60)

model.eval()
running_sum = None
n_frames = 0
t0 = time.time()

CHUNK_SIZE = 8  # small chunks to limit VRAM on extract

with av.open(out_path) as container:
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    chunk_buf = []

    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(rgb).permute(2, 0, 1)  # uint8 [3, H, W]
        chunk_buf.append(t)

        if len(chunk_buf) >= CHUNK_SIZE:
            video_chunk = torch.stack(chunk_buf).to(device).float().div_(255.0)
            chunk_buf.clear()
            with torch.no_grad():
                outputs = model.detect(video_chunk, is_video=True)
            preds = outputs["preds"]
            if preds.dim() == 4:
                preds = preds.mean(dim=(-2, -1))
            bit_preds = preds[:, 1:].cpu()
            chunk_sum = (bit_preds * bit_preds.abs()).sum(dim=0)
            if running_sum is None:
                running_sum = chunk_sum
            else:
                running_sum.add_(chunk_sum)
            n_frames += bit_preds.shape[0]
            del video_chunk, outputs, preds, bit_preds, chunk_sum
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            mem = torch.cuda.memory_allocated(device) / 1024**2
            print(f"  extracted {n_frames}f  ({elapsed:.1f}s, {n_frames/elapsed:.0f}f/s)  [VRAM={mem:.0f}MB]")

            if n_frames >= MAX_FRAMES:
                break

    # Flush remaining
    if chunk_buf and n_frames < MAX_FRAMES:
        video_chunk = torch.stack(chunk_buf).to(device).float().div_(255.0)
        chunk_buf.clear()
        with torch.no_grad():
            outputs = model.detect(video_chunk, is_video=True)
        preds = outputs["preds"]
        if preds.dim() == 4:
            preds = preds.mean(dim=(-2, -1))
        bit_preds = preds[:, 1:].cpu()
        chunk_sum = (bit_preds * bit_preds.abs()).sum(dim=0)
        if running_sum is None:
            running_sum = chunk_sum
        else:
            running_sum.add_(chunk_sum)
        n_frames += bit_preds.shape[0]
        del video_chunk, outputs, preds, bit_preds, chunk_sum

t_extract = time.time() - t0

# Aggregate: running squared_avg
decoded_msg = running_sum / n_frames  # [K]
msg = (decoded_msg > 0).unsqueeze(0)  # [1, K]
decode = msg_tensor_to_text_bch(msg)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("RESULTS")
print("=" * 60)
print(f"  Embed:          {t_embed:.1f}s ({result['total_frames']}f, {result['total_frames']/t_embed:.0f}f/s)")
print(f"  Extract:        {t_extract:.1f}s ({n_frames}f, {n_frames/t_extract:.0f}f/s)")
print(f"  Output size:    {out_size:.1f}MB")
print(f"  VRAM peak:      {torch.cuda.max_memory_allocated(device)/1024**2:.0f}MB")
print(f"  ---")
print(f"  WM Detected:    {decode['correctable']}")
print(f"  Embed text:     '{WM_TEXT}'")
print(f"  Extracted text: '{decode['decoded_text']}'")
print(f"  Match:          {'YES' if decode.get('decoded_text') == WM_TEXT else 'NO'}")
print(f"  Bits (first 20): {decode.get('bits_list', [])[:20]}")
print("=" * 60)

# Cleanup
os.unlink(out_path)
torch.cuda.empty_cache()
