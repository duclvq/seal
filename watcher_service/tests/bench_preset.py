"""
Quick benchmark: compare NVENC presets (p1 fastest → p7 slowest).
Usage: python bench_preset.py --input ../assets/videos/1.mp4
"""
import argparse, os, sys, tempfile, time
from pathlib import Path

import av
import numpy as np

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def bench_encode(input_path: str, presets=("p1", "p2", "p4", "p7")):
    """Decode all frames, then encode with each preset and measure fps."""
    # Decode frames first
    frames = []
    with av.open(input_path) as c:
        vs = c.streams.video[0]
        vs.thread_type = "FRAME"
        w, h = vs.width, vs.height
        fps = float(vs.average_rate or vs.guessed_rate or 24)
        for f in c.decode(vs):
            frames.append(f.to_ndarray(format="rgb24"))
    n = len(frames)
    print(f"Decoded {n} frames ({w}x{h}) @ {fps:.1f}fps")

    # Check NVENC
    try:
        av.codec.Codec("h264_nvenc", "w")
        has_nvenc = True
    except Exception:
        has_nvenc = False
        print("NVENC not available, testing libx264 only")

    results = []
    for preset in presets:
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        codec = "h264_nvenc" if has_nvenc else "h264"
        opts = {"preset": preset} if has_nvenc else {"preset": "ultrafast"}

        out = av.open(tmp, mode="w")
        s = out.add_stream(codec, rate=int(fps))
        H = h - h % 2
        W = w - w % 2
        s.width = W
        s.height = H
        s.pix_fmt = "yuv420p"
        s.options = opts

        t0 = time.perf_counter()
        _encode = s.encode
        _mux = out.mux
        _from_ndarray = av.VideoFrame.from_ndarray
        for frame_np in frames:
            for pkt in _encode(_from_ndarray(frame_np[:H, :W], format="rgb24")):
                _mux(pkt)
        for pkt in _encode():
            _mux(pkt)
        out.close()
        dt = time.perf_counter() - t0

        sz = os.path.getsize(tmp) / 1024 / 1024
        enc_fps = n / dt
        results.append((preset, enc_fps, sz, dt))
        print(f"  {preset}: {enc_fps:.1f} fps  ({dt:.2f}s)  size={sz:.1f}MB")
        os.unlink(tmp)

    if not has_nvenc:
        return results

    # Also test libx264 ultrafast for comparison
    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out = av.open(tmp, mode="w")
    s = out.add_stream("h264", rate=int(fps))
    s.width = W
    s.height = H
    s.pix_fmt = "yuv420p"
    s.options = {"preset": "ultrafast", "crf": "18"}
    t0 = time.perf_counter()
    for frame_np in frames:
        for pkt in s.encode(av.VideoFrame.from_ndarray(frame_np[:H, :W], format="rgb24")):
            out.mux(pkt)
    for pkt in s.encode():
        out.mux(pkt)
    out.close()
    dt = time.perf_counter() - t0
    sz = os.path.getsize(tmp) / 1024 / 1024
    print(f"  libx264-ultrafast: {n/dt:.1f} fps  ({dt:.2f}s)  size={sz:.1f}MB")
    os.unlink(tmp)

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()
    bench_encode(args.input)
