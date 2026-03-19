"""
Debug: compare first N frames of input vs output to find visual artifacts.
Computes per-frame PSNR and frame-to-frame difference to detect jumps.
"""
import sys, os
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import av
import numpy as np

INPUT  = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
OUTPUT = str(_SERVICE_DIR / "data" / "test_output" / "DJI_0152_wm.mp4")
N_FRAMES = 200  # analyze first 200 frames (4s at 50fps)

def extract_frames(path, n):
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        for i, f in enumerate(c.decode(s)):
            if i >= n:
                break
            frames.append(f.to_ndarray(format="rgb24"))
    return frames

print(f"Extracting {N_FRAMES} frames from input...")
inp_frames = extract_frames(INPUT, N_FRAMES)
print(f"Extracting {N_FRAMES} frames from output...")
out_frames = extract_frames(OUTPUT, N_FRAMES)

n = min(len(inp_frames), len(out_frames))
print(f"\nGot {len(inp_frames)} input, {len(out_frames)} output frames\n")

# 1. Per-frame PSNR (input vs output) — detects watermark strength variation
print("=== Per-frame PSNR (input vs output) ===")
print("Frame  PSNR(dB)  MSE")
psnrs = []
for i in range(n):
    diff = inp_frames[i].astype(np.float32) - out_frames[i].astype(np.float32)
    mse = np.mean(diff ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99.0
    psnrs.append(psnr)
    if i < 100 or i % 32 == 0 or i % 32 == 31:  # chunk boundaries
        print(f"  {i:4d}  {psnr:6.2f}    {mse:.2f}")

# 2. Frame-to-frame difference in OUTPUT — detects temporal jumps
print("\n=== Frame-to-frame diff in OUTPUT (consecutive frames) ===")
print("Frame  MAE(vs prev)  Jump?")
prev = out_frames[0].astype(np.float32)
maes = []
for i in range(1, n):
    cur = out_frames[i].astype(np.float32)
    mae = np.mean(np.abs(cur - prev))
    maes.append(mae)
    is_jump = ""
    if i > 1 and len(maes) > 2:
        avg_mae = np.mean(maes[max(0,len(maes)-10):len(maes)-1])
        if mae > avg_mae * 3:
            is_jump = " <<<< JUMP"
    if i < 100 or i % 32 == 0 or i % 32 == 31 or is_jump:
        print(f"  {i:4d}  {mae:8.3f}     {is_jump}")
    prev = cur

# 3. Chunk boundary analysis
print("\n=== Chunk boundary PSNR drops ===")
chunk_size = 32
for boundary in range(chunk_size, n, chunk_size):
    if boundary < n:
        before = psnrs[boundary - 1]
        after = psnrs[boundary]
        delta = after - before
        flag = " <<<< DROP" if abs(delta) > 1.0 else ""
        print(f"  Boundary {boundary-1}->{boundary}: PSNR {before:.2f} -> {after:.2f} (Δ={delta:+.2f}){flag}")

print("\nDone.")
