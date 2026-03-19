"""
Debug: check if output has fewer frames than input (frame dropping)
and check frame-by-frame Y-plane comparison more carefully.
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

INPUT = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
OUTPUT = str(_SERVICE_DIR / "data" / "test_output" / "DJI_0152_wm.mp4")

print("=== Frame count ===")
with av.open(INPUT) as c:
    s = c.streams.video[0]
    print(f"Input: {s.frames} frames, {s.width}x{s.height}")

with av.open(OUTPUT) as c:
    s = c.streams.video[0]
    print(f"Output: {s.frames} frames, {s.width}x{s.height}")
    if s.frames < 3519:
        print(f"*** MISSING {3519 - s.frames} frames!")

# Check: does the first debug script compare frames correctly?
# The issue might be that output has 3517 frames vs input 3519
# meaning 2 frames are dropped, causing frame misalignment

# Let's check by comparing Y-plane only (no chroma issue)
print("\n=== Y-plane PSNR (input vs output) - first 40 frames ===")
print("Frame  PSNR_Y(dB)  MSE_Y")

inp_frames_y = []
with av.open(INPUT) as c:
    s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i >= 40: break
        nd = f.to_ndarray(format="yuv420p")
        H = (nd.shape[0] * 2) // 3
        inp_frames_y.append(nd[:H].copy())

out_frames_y = []
with av.open(OUTPUT) as c:
    s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i >= 40: break
        nd = f.to_ndarray(format="yuv420p")
        H = (nd.shape[0] * 2) // 3
        out_frames_y.append(nd[:H].copy())

for i in range(min(len(inp_frames_y), len(out_frames_y))):
    diff = inp_frames_y[i].astype(np.float32) - out_frames_y[i].astype(np.float32)
    mse = np.mean(diff ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99
    print(f"  {i:4d}  {psnr:8.2f}  {mse:8.2f}")

# Check if frame alignment is off
print("\n=== Cross-correlation: is output shifted? ===")
# Compare input[20] with output[20], output[19], output[21]
for offset in [-2, -1, 0, 1, 2]:
    i_in = 20
    i_out = 20 + offset
    if 0 <= i_out < len(out_frames_y):
        diff = inp_frames_y[i_in].astype(np.float32) - out_frames_y[i_out].astype(np.float32)
        mse = np.mean(diff ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99
        print(f"  input[{i_in}] vs output[{i_out}]: PSNR={psnr:.2f} MSE={mse:.2f}")

print("\nDone.")
