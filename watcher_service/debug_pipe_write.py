"""
Debug: simulate exactly what _enc_worker does — write YUV to FFmpeg pipe,
then read back and compare with input to find where corruption happens.
Write only 32 frames (1 chunk) of UNMODIFIED input to isolate encode issue.
"""
import sys, os, subprocess, threading
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
TMP_OUT = str(_SERVICE_DIR / "data" / "test_output" / "debug_pipe_32f.mp4")

# Extract 32 frames as YUV420P packed [H*3//2, W]
print("Extracting 32 input frames (YUV420P packed)...")
frames = []
with av.open(INPUT) as c:
    s = c.streams.video[0]
    fps = float(s.average_rate or s.guessed_rate)
    for i, f in enumerate(c.decode(s)):
        if i >= 32: break
        frames.append(f.to_ndarray(format="yuv420p"))

H_full = (frames[0].shape[0] * 2) // 3
W = frames[0].shape[1]
H = H_full
print(f"Frame shape: {frames[0].shape}, H={H}, W={W}, fps={fps}")

# Method 1: Write packed format directly (what _enc_worker does)
print("\n=== Method 1: Write packed [H*3//2, W] tobytes directly ===")
cmd = [
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
    "-f", "rawvideo", "-pix_fmt", "yuv420p",
    "-s", f"{W}x{H}", "-r", str(int(fps)),
    "-i", "pipe:0",
    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    "-pix_fmt", "yuv420p", TMP_OUT,
]
print(f"CMD: {' '.join(cmd)}")

stderr_lines = []
def drain(proc, buf):
    for line in proc.stderr:
        buf.append(line.decode(errors="replace").rstrip())

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
t = threading.Thread(target=drain, args=(proc, stderr_lines), daemon=True)
t.start()

for i, frame in enumerate(frames):
    raw = frame.tobytes()
    proc.stdin.write(raw)
    if i == 0:
        print(f"  Frame 0: wrote {len(raw)} bytes")

proc.stdin.close()
proc.wait()
t.join(timeout=5)
if stderr_lines:
    print(f"  FFmpeg stderr: {stderr_lines[-3:]}")
print(f"  FFmpeg rc={proc.returncode}")

# Read back and compare
print("\n=== Read back and compare Y-plane ===")
out_frames = []
with av.open(TMP_OUT) as c:
    s = c.streams.video[0]
    print(f"  Output: {s.width}x{s.height}, {s.frames} frames")
    for i, f in enumerate(c.decode(s)):
        if i >= 32: break
        out_frames.append(f.to_ndarray(format="yuv420p"))

print(f"  Got {len(out_frames)} frames back")
print("\nFrame  PSNR_Y  MSE_Y")
for i in range(min(len(frames), len(out_frames))):
    inp_y = frames[i][:H].astype(np.float32)
    out_y = out_frames[i][:H].astype(np.float32)
    mse = np.mean((inp_y - out_y) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99
    print(f"  {i:4d}  {psnr:7.2f}  {mse:7.2f}")

# Method 2: Write PROPER planar format
print("\n\n=== Method 2: Write proper planar YUV420P ===")
TMP_OUT2 = str(_SERVICE_DIR / "data" / "test_output" / "debug_pipe_32f_planar.mp4")
hh, hw = H // 2, W // 2

proc2 = subprocess.Popen(
    ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
     "-f", "rawvideo", "-pix_fmt", "yuv420p",
     "-s", f"{W}x{H}", "-r", str(int(fps)),
     "-i", "pipe:0",
     "-c:v", "libx264", "-preset", "fast", "-crf", "18",
     "-pix_fmt", "yuv420p", TMP_OUT2],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
)
stderr2 = []
t2 = threading.Thread(target=drain, args=(proc2, stderr2), daemon=True)
t2.start()

for i, frame in enumerate(frames):
    Y = frame[:H]
    Cb = frame[H:, :hw]
    Cr = frame[H:, hw:]
    raw = Y.tobytes() + Cb.tobytes() + Cr.tobytes()
    proc2.stdin.write(raw)
    if i == 0:
        print(f"  Frame 0: wrote {len(raw)} bytes (Y={H*W} + Cb={hh*hw} + Cr={hh*hw})")

proc2.stdin.close()
proc2.wait()
t2.join(timeout=5)
if stderr2:
    print(f"  FFmpeg stderr: {stderr2[-3:]}")
print(f"  FFmpeg rc={proc2.returncode}")

# Read back method 2
print("\n=== Read back method 2 (planar) ===")
out_frames2 = []
with av.open(TMP_OUT2) as c:
    s = c.streams.video[0]
    print(f"  Output: {s.width}x{s.height}, {s.frames} frames")
    for i, f in enumerate(c.decode(s)):
        if i >= 32: break
        out_frames2.append(f.to_ndarray(format="yuv420p"))

print(f"  Got {len(out_frames2)} frames back")
print("\nFrame  PSNR_Y  MSE_Y")
for i in range(min(len(frames), len(out_frames2))):
    inp_y = frames[i][:H].astype(np.float32)
    out_y = out_frames2[i][:H].astype(np.float32)
    mse = np.mean((inp_y - out_y) ** 2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else 99
    print(f"  {i:4d}  {psnr:7.2f}  {mse:7.2f}")

print("\nDone.")
