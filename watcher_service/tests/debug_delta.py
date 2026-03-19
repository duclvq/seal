"""
Debug: inspect the watermark delta values per frame in the first chunk.
Check if delta magnitude is consistent across all 32 frames.
"""
import sys, os
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent
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

import av
import numpy as np
import torch
from worker import load_video_model, _get_watermark_delta, _encode_audio_msg
from worker import _yuv420p_to_rgb_gpu, _yuv420p_resize_down_gpu

CKPT  = "output/run2_video/checkpoint350.pth"
INPUT = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
device = torch.device("cuda")

print("Loading model...")
model = load_video_model(CKPT, device)
print(f"model.chunk_size={model.chunk_size}, step_size={model.step_size}, "
      f"video_mode={model.video_mode}")
print(f"scaling_i={model.blender.scaling_i}, scaling_w={model.blender.scaling_w}")
print(f"embedder.yuv={model.embedder.yuv}")

# Extract first 64 frames as YUV420P
print("\nExtracting first 64 frames...")
frames_yuv = []
with av.open(INPUT) as c:
    s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i >= 64:
            break
        frames_yuv.append(f.to_ndarray(format="yuv420p"))

print(f"Got {len(frames_yuv)} frames, shape={frames_yuv[0].shape}")

# Stack into tensor
yuv_batch = torch.from_numpy(np.stack(frames_yuv))  # [64, H*3//2, W]
N, total_h, W = yuv_batch.shape
H = (total_h * 2) // 3
print(f"YUV batch: {yuv_batch.shape}, H={H}, W={W}")

# Process chunk 1 (frames 0-31)
chunk1 = yuv_batch[:32].to(device)
yuv_256_1 = _yuv420p_resize_down_gpu(chunk1, 256, 256)
rgb_256_1 = _yuv420p_to_rgb_gpu(yuv_256_1, device).float().div_(255.0)
del yuv_256_1

print(f"\nChunk 1 rgb_256: {rgb_256_1.shape}")

# Get watermark delta
msg_gpu = _encode_audio_msg("test", device)
delta_1 = _get_watermark_delta(rgb_256_1, model, msg_gpu, device, keep_on_gpu=True)
print(f"Delta 1: {delta_1.shape}")

# Analyze delta magnitude per frame
print("\n=== Delta magnitude per frame (chunk 1) ===")
print("Frame  mean_abs   max_abs   std")
for i in range(32):
    d = delta_1[i]
    print(f"  {i:4d}  {d.abs().mean():.6f}  {d.abs().max():.4f}  {d.std():.6f}")

# Now simulate the blend like _gpu_worker does
scaling_w = float(model.blender.scaling_w)
scaling_i = float(model.blender.scaling_i)
print(f"\nscaling_w={scaling_w}, scaling_i={scaling_i}")
print(f"delta * scaling_w * 255 range:")

delta_scaled = delta_1 * scaling_w * 255.0
for i in range(32):
    d = delta_scaled[i]
    print(f"  Frame {i:4d}: mean_abs={d.abs().mean():.2f}  max_abs={d.abs().max():.2f}  "
          f"(would shift Y by ±{d.abs().mean():.1f} levels)")

# Check if model has attenuation and what it does
if model.attenuation is not None:
    print("\n=== Attenuation heatmaps ===")
    model.attenuation.to(device)
    hmaps = model.attenuation.heatmaps(rgb_256_1.to(device))
    print(f"Heatmap shape: {hmaps.shape}")
    for i in range(min(32, hmaps.shape[0])):
        h = hmaps[i]
        print(f"  Frame {i:4d}: mean={h.mean():.4f}  min={h.min():.4f}  max={h.max():.4f}")

torch.cuda.empty_cache()
print("\nDone.")
