"""
Debug: write raw YUV frames to file, then check if FFmpeg reads them correctly.
This isolates whether the issue is in YUV packing or FFmpeg encoding.
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

INPUT = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
OUTPUT = str(_SERVICE_DIR / "data" / "test_output" / "DJI_0152_wm.mp4")

# Check output video metadata
print("=== Output video info ===")
with av.open(OUTPUT) as c:
    s = c.streams.video[0]
    print(f"  codec: {s.codec_context.name}")
    print(f"  size: {s.width}x{s.height}")
    print(f"  pix_fmt: {s.codec_context.pix_fmt}")
    print(f"  fps: {float(s.average_rate or s.guessed_rate)}")
    print(f"  frames: {s.frames}")
    print(f"  duration: {float(s.duration * s.time_base):.2f}s")

# Check input video
print("\n=== Input video info ===")
with av.open(INPUT) as c:
    s = c.streams.video[0]
    print(f"  codec: {s.codec_context.name}")
    print(f"  size: {s.width}x{s.height}")
    print(f"  pix_fmt: {s.codec_context.pix_fmt}")
    print(f"  fps: {float(s.average_rate or s.guessed_rate)}")
    print(f"  frames: {s.frames}")

# Check the YUV frame sizes match
print("\n=== Frame size check ===")
with av.open(INPUT) as c:
    s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i >= 1: break
        nd = f.to_ndarray(format="yuv420p")
        print(f"  Input YUV420P frame: {nd.shape}")
        H = (nd.shape[0] * 2) // 3
        W = nd.shape[1]
        print(f"  H={H}, W={W}")
        print(f"  H%2={H%2}, W%2={W%2}")
        # Check what FFmpeg would expect
        H_enc = H - H % 2
        W_enc = W - W % 2
        print(f"  H_enc={H_enc}, W_enc={W_enc}")
        if H_enc != H or W_enc != W:
            print(f"  *** MISMATCH: FFmpeg uses {W_enc}x{H_enc} but frames are {W}x{H}")
            print(f"  *** This means {H-H_enc} rows and {W-W_enc} cols are LOST")

# Now check: does the enc_worker write correct raw bytes?
# The enc_worker converts YUV packed [H*3//2, W] to raw bytes via .tobytes()
# FFmpeg expects planar YUV420P: Y plane (H*W) + U plane (H/2*W/2) + V plane (H/2*W/2)
# But our packed format is [H*3//2, W] where:
#   rows 0..H-1 = Y
#   rows H..H*3//2-1 = interleaved Cb|Cr (side by side)
# FFmpeg expects: YYYYYYYY...UUU...VVV... (planar)
# Our format:    YYYYYYYY...[Cb|Cr][Cb|Cr]... (packed side-by-side)

print("\n=== YUV packing format analysis ===")
with av.open(INPUT) as c:
    s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i >= 1: break
        nd = f.to_ndarray(format="yuv420p")
        H = (nd.shape[0] * 2) // 3
        W = nd.shape[1]
        hh, hw = H // 2, W // 2
        
        print(f"  Full array shape: {nd.shape}")
        print(f"  Y plane: rows 0:{H} = {nd[:H].shape}")
        print(f"  Chroma rows: {H}:{nd.shape[0]} = {nd[H:].shape}")
        
        chroma = nd[H:]
        print(f"  Chroma block shape: {chroma.shape}")
        print(f"  Expected Cb: [{hh}, {hw}], Cr: [{hh}, {hw}]")
        
        # PyAV yuv420p to_ndarray returns [H*3//2, W] where:
        # Y = [0:H, 0:W]
        # U = [H:H+H//2, 0:W//2]  
        # V = [H:H+H//2, W//2:W]
        # This is NOT standard planar YUV420P!
        
        # FFmpeg rawvideo -pix_fmt yuv420p expects:
        # Y plane: H*W bytes
        # U plane: (H/2)*(W/2) bytes  
        # V plane: (H/2)*(W/2) bytes
        # Total: H*W*3/2 bytes
        
        expected_bytes = H * W + 2 * (hh * hw)
        actual_bytes = nd.nbytes
        tobytes_len = len(nd.tobytes())
        
        print(f"\n  Expected planar YUV420P bytes: {expected_bytes}")
        print(f"  Actual ndarray bytes: {actual_bytes}")
        print(f"  tobytes() length: {tobytes_len}")
        
        if actual_bytes != expected_bytes:
            print(f"  *** SIZE MISMATCH!")
            print(f"  *** ndarray is {nd.shape[0]}*{nd.shape[1]} = {nd.shape[0]*nd.shape[1]} bytes")
            print(f"  *** but planar YUV420P needs {expected_bytes} bytes")
            
        # Check: when we do tobytes() on [H*3//2, W], does it produce
        # correct planar layout or wrong interleaved layout?
        raw = nd.tobytes()
        
        # If correct planar: first H*W bytes = Y, next hh*hw = U, next hh*hw = V
        # If our packed format: first H*W bytes = Y, then hh*W bytes = [Cb|Cr] interleaved
        
        # Let's check by comparing U plane
        y_end = H * W
        u_end = y_end + hh * hw
        
        # From tobytes (row-major): Y is rows 0..H-1, then chroma rows
        # Chroma rows are [H:H+hh, 0:W] which is [Cb_row | Cr_row] side by side
        # tobytes() will give: Cb_row0 + Cr_row0 + Cb_row1 + Cr_row1 + ...
        # But FFmpeg expects: Cb_row0 + Cb_row1 + ... + Cr_row0 + Cr_row1 + ...
        
        print(f"\n  *** CRITICAL: tobytes() on packed [H*3//2, W] array produces")
        print(f"  *** interleaved chroma: [Cb_row0|Cr_row0][Cb_row1|Cr_row1]...")
        print(f"  *** But FFmpeg -pix_fmt yuv420p expects PLANAR:")
        print(f"  *** [Y plane][U plane][V plane]")
        print(f"  *** This means chroma planes are WRONG in the FFmpeg pipe!")
        
        # Verify: extract what FFmpeg would interpret as U and V
        raw_u = np.frombuffer(raw[y_end:u_end], dtype=np.uint8).reshape(hh, hw)
        raw_v = np.frombuffer(raw[u_end:u_end+hh*hw], dtype=np.uint8).reshape(hh, hw)
        
        # What the actual U and V should be
        real_u = nd[H:, :hw]  # left half of chroma block
        real_v = nd[H:, hw:]  # right half of chroma block
        
        # Compare
        u_match = np.array_equal(raw_u, real_u)
        v_match = np.array_equal(raw_v, real_v)
        print(f"\n  U plane from tobytes matches real U: {u_match}")
        print(f"  V plane from tobytes matches real V: {v_match}")
        
        if not u_match:
            print(f"  U diff: mean={np.mean(np.abs(raw_u.astype(float)-real_u.astype(float))):.2f}")
        if not v_match:
            print(f"  V diff: mean={np.mean(np.abs(raw_v.astype(float)-real_v.astype(float))):.2f}")

print("\nDone.")
