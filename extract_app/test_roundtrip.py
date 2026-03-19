"""
Round-trip test: Embed key 'vtvwm0' → Extract → Verify.
Tests embed pipeline + extract web app logic end-to-end.
"""
import os, sys, tempfile, time
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
from worker import load_video_model, embed_to_file
from web_demo.core.ecc import msg_tensor_to_text_bch

CKPT = "output/run2_video/checkpoint350.pth"
INPUT = r"D:\samu_videos_gpu\DJI_20260104092139_0153_D_gpu_hevc.mp4"
WM_TEXT = "vtv0"  # BCH-256 max 4 ASCII bytes
EXTRACT_FRAMES = 600
CHUNK = 8

device = torch.device("cuda")

# ═══ EMBED ═══
print("=" * 60)
print(f"EMBED: key='{WM_TEXT}'")
print("=" * 60)

model = load_video_model(CKPT, device)

fd, out_path = tempfile.mkstemp(suffix=".mp4")
os.close(fd)

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
out_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"  Done: {t_embed:.1f}s, {result['total_frames']}f, "
      f"{result['total_frames']/t_embed:.0f}f/s, {out_mb:.1f}MB")
print(f"  Mode: {result['mode']}")

# ═══ EXTRACT (streaming, same logic as app.py) ═══
print(f"\n{'=' * 60}")
print(f"EXTRACT: streaming chunk={CHUNK}, max={EXTRACT_FRAMES}f")
print("=" * 60)

model.eval()
running_sum = None
n_frames = 0
t0 = time.time()

with av.open(out_path) as container:
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    chunk_buf = []

    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(rgb).permute(2, 0, 1)  # uint8
        chunk_buf.append(t)

        if len(chunk_buf) >= CHUNK:
            video_chunk = torch.stack(chunk_buf).to(device).float().div_(255.0)
            chunk_buf.clear()
            with torch.no_grad():
                outputs = model.detect(video_chunk, is_video=True)
            preds = outputs["preds"]
            if preds.dim() == 4:
                preds = preds.mean(dim=(-2, -1))
            bp = preds[:, 1:].cpu()
            cs = (bp * bp.abs()).sum(dim=0)
            running_sum = cs if running_sum is None else running_sum + cs
            n_frames += bp.shape[0]
            del video_chunk, outputs, preds, bp, cs
            torch.cuda.empty_cache()

        if n_frames >= EXTRACT_FRAMES:
            break

    if chunk_buf:
        video_chunk = torch.stack(chunk_buf).to(device).float().div_(255.0)
        chunk_buf.clear()
        with torch.no_grad():
            outputs = model.detect(video_chunk, is_video=True)
        preds = outputs["preds"]
        if preds.dim() == 4:
            preds = preds.mean(dim=(-2, -1))
        bp = preds[:, 1:].cpu()
        cs = (bp * bp.abs()).sum(dim=0)
        running_sum = cs if running_sum is None else running_sum + cs
        n_frames += bp.shape[0]
        del video_chunk, outputs, preds, bp, cs

t_extract = time.time() - t0
decoded_msg = running_sum / n_frames
msg = (decoded_msg > 0).unsqueeze(0)
decode = msg_tensor_to_text_bch(msg)

# ═══ RESULTS ═══
print(f"\n{'=' * 60}")
print("RESULTS")
print("=" * 60)
print(f"  Embed key:      '{WM_TEXT}'")
print(f"  Embed:          {t_embed:.1f}s ({result['total_frames']}f, {result['total_frames']/t_embed:.0f}f/s)")
print(f"  Extract:        {t_extract:.1f}s ({n_frames}f, {n_frames/t_extract:.0f}f/s)")
print(f"  VRAM:           {torch.cuda.max_memory_allocated(device)/1024**2:.0f}MB peak")
print(f"  ---")
print(f"  WM Detected:    {decode['correctable']}")
print(f"  Extracted text: '{decode['decoded_text']}'")
match = decode.get('decoded_text') == WM_TEXT
print(f"  Match:          {'YES' if match else 'NO'}")
if not match:
    print(f"  Expected bits:  (first 20 of '{WM_TEXT}')")
print(f"  Bits (first 20): {decode.get('bits_list', [])[:20]}")
print("=" * 60)

# Cleanup
os.unlink(out_path)
torch.cuda.empty_cache()
