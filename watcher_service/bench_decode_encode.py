"""
Benchmark decode & encode bottlenecks — so sánh các phương pháp tối ưu.

Tests:
  1. Decode: FRAME vs AUTO thread_type, thread_count 2/4/8
  2. Decode: to_ndarray("rgb24") vs to_ndarray("yuv420p") + manual convert
  3. Decode: numpy copy overhead (ndarray → torch tensor)
  4. Encode: libx264 preset fast vs ultrafast vs veryfast
  5. Encode: NVENC presets p1/p2/p4
  6. Encode: single-frame vs batch-write
  7. Encode: encoder thread_count
  8. Mux overhead: ffmpeg subprocess vs PyAV stream copy

Usage:
    python bench_decode_encode.py --input ../data/val/gi-diploma-66013582898105388892353.mp4
    python bench_decode_encode.py --input ../data/val/gi-diploma-66013582898105388892353.mp4 --frames 600
"""
import argparse
import gc
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import av
import numpy as np
import torch

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _tmpfile(suffix=".mp4"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# ═════════════════════════════════════════════════════════════════════════════
# DECODE benchmarks
# ═════════════════════════════════════════════════════════════════════════════

def bench_decode_thread_modes(path: str, max_frames: int, runs: int = 3):
    """Compare thread_type FRAME vs AUTO, and different thread_count values."""
    log.info(f"\n{'='*70}")
    log.info("DECODE: thread_type × thread_count")
    log.info(f"{'='*70}")

    configs = [
        ("FRAME", 2), ("FRAME", 4), ("FRAME", 8),
        ("AUTO", 2),  ("AUTO", 4),  ("AUTO", 8),
    ]

    results = []
    for ttype, tcount in configs:
        times = []
        n = 0
        for _ in range(runs):
            gc.collect()
            t0 = time.perf_counter()
            with av.open(path) as c:
                s = c.streams.video[0]
                s.thread_type = ttype
                s.codec_context.thread_count = tcount
                count = 0
                for frame in c.decode(s):
                    _ = frame.to_ndarray(format="rgb24")
                    count += 1
                    if count >= max_frames:
                        break
            dt = time.perf_counter() - t0
            times.append(dt)
            n = count

        avg = sum(times) / len(times)
        fps = n / avg
        results.append((ttype, tcount, avg, fps))
        log.info(f"  {ttype:5s} threads={tcount}:  {avg:.2f}s  {fps:.0f} fps  "
                 f"[{', '.join(f'{t:.2f}' for t in times)}]")

    best = max(results, key=lambda x: x[3])
    log.info(f"  → Best: {best[0]} threads={best[1]} ({best[3]:.0f} fps)")
    return results


def bench_decode_format_conversion(path: str, max_frames: int, runs: int = 3):
    """Compare to_ndarray format: rgb24 vs nv12 + manual conversion."""
    log.info(f"\n{'='*70}")
    log.info("DECODE: frame format conversion overhead")
    log.info(f"{'='*70}")

    # Method 1: to_ndarray("rgb24") — current approach
    times_rgb = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        with av.open(path) as c:
            s = c.streams.video[0]
            s.thread_type = "FRAME"
            count = 0
            for frame in c.decode(s):
                arr = frame.to_ndarray(format="rgb24")
                t = torch.from_numpy(arr).permute(2, 0, 1)
                count += 1
                if count >= max_frames:
                    break
        times_rgb.append(time.perf_counter() - t0)

    # Method 2: to_ndarray("rgb24") with contiguous copy
    times_contig = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        with av.open(path) as c:
            s = c.streams.video[0]
            s.thread_type = "FRAME"
            count = 0
            for frame in c.decode(s):
                arr = frame.to_ndarray(format="rgb24")
                t = torch.as_tensor(arr).permute(2, 0, 1).contiguous()
                count += 1
                if count >= max_frames:
                    break
        times_contig.append(time.perf_counter() - t0)

    # Method 3: raw decode only (no conversion) — baseline
    times_raw = []
    for _ in range(runs):
        gc.collect()
        t0 = time.perf_counter()
        with av.open(path) as c:
            s = c.streams.video[0]
            s.thread_type = "FRAME"
            count = 0
            for frame in c.decode(s):
                # Just access the frame, no ndarray conversion
                _ = frame.width
                count += 1
                if count >= max_frames:
                    break
        times_raw.append(time.perf_counter() - t0)

    n = max_frames
    avg_rgb = sum(times_rgb) / len(times_rgb)
    avg_contig = sum(times_contig) / len(times_contig)
    avg_raw = sum(times_raw) / len(times_raw)

    log.info(f"  Raw decode (no convert):     {avg_raw:.2f}s  {n/avg_raw:.0f} fps")
    log.info(f"  + to_ndarray(rgb24) + torch: {avg_rgb:.2f}s  {n/avg_rgb:.0f} fps  "
             f"(+{(avg_rgb-avg_raw)/avg_raw*100:.0f}% overhead)")
    log.info(f"  + as_tensor + contiguous:    {avg_contig:.2f}s  {n/avg_contig:.0f} fps  "
             f"(+{(avg_contig-avg_raw)/avg_raw*100:.0f}% overhead)")
    log.info(f"  → Conversion overhead: {avg_rgb - avg_raw:.2f}s "
             f"({(avg_rgb-avg_raw)/avg_rgb*100:.0f}% of total decode time)")


def bench_decode_batch_stack(path: str, max_frames: int, chunk_size: int = 45):
    """Measure torch.stack overhead for chunked decoding."""
    log.info(f"\n{'='*70}")
    log.info(f"DECODE: torch.stack overhead (chunk={chunk_size})")
    log.info(f"{'='*70}")

    # Decode frames first
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "FRAME"
        for f in c.decode(s):
            frames.append(torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1))
            if len(frames) >= max_frames:
                break

    n = len(frames)
    H, W = frames[0].shape[1], frames[0].shape[2]
    frame_mb = 3 * H * W / 1024**2

    # Method 1: torch.stack (current)
    gc.collect()
    t0 = time.perf_counter()
    for i in range(0, n, chunk_size):
        chunk = frames[i:i+chunk_size]
        batch = torch.stack(chunk).float() / 255.0
        del batch
    t_stack = time.perf_counter() - t0

    # Method 2: pre-allocated tensor + copy
    gc.collect()
    t0 = time.perf_counter()
    for i in range(0, n, chunk_size):
        chunk = frames[i:i+chunk_size]
        cs = len(chunk)
        batch = torch.empty(cs, 3, H, W, dtype=torch.float32)
        for j, f in enumerate(chunk):
            batch[j] = f.float() / 255.0
        del batch
    t_prealloc = time.perf_counter() - t0

    # Method 3: torch.stack + div_ in-place
    gc.collect()
    t0 = time.perf_counter()
    for i in range(0, n, chunk_size):
        chunk = frames[i:i+chunk_size]
        batch = torch.stack(chunk).float().div_(255.0)
        del batch
    t_inplace = time.perf_counter() - t0

    n_chunks = (n + chunk_size - 1) // chunk_size
    log.info(f"  {n} frames ({W}x{H}), {n_chunks} chunks, {frame_mb:.1f} MB/frame")
    log.info(f"  torch.stack + /255:     {t_stack:.3f}s")
    log.info(f"  pre-alloc + copy:       {t_prealloc:.3f}s  "
             f"({'faster' if t_prealloc < t_stack else 'slower'} "
             f"{abs(t_prealloc-t_stack)/t_stack*100:.0f}%)")
    log.info(f"  torch.stack + div_():   {t_inplace:.3f}s  "
             f"({'faster' if t_inplace < t_stack else 'slower'} "
             f"{abs(t_inplace-t_stack)/t_stack*100:.0f}%)")

    del frames
    gc.collect()



# ═════════════════════════════════════════════════════════════════════════════
# ENCODE benchmarks
# ═════════════════════════════════════════════════════════════════════════════

def _decode_frames_np(path: str, max_frames: int) -> tuple:
    """Decode frames as numpy uint8 array [N,H,W,3]. Returns (array, fps)."""
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "FRAME"
        fps = float(s.average_rate or s.guessed_rate or 25)
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    return np.stack(frames), fps


def bench_encode_presets(frames_np: np.ndarray, fps: float):
    """Compare libx264 presets and NVENC presets."""
    log.info(f"\n{'='*70}")
    log.info("ENCODE: codec × preset comparison")
    log.info(f"{'='*70}")

    N, H, W, C = frames_np.shape
    H2 = H - H % 2
    W2 = W - W % 2
    log.info(f"  {N} frames, {W2}x{H2}")

    configs = [
        # (label, codec, options)
        ("x264 ultrafast CRF18", "h264", {"preset": "ultrafast", "crf": "18"}),
        ("x264 veryfast  CRF18", "h264", {"preset": "veryfast", "crf": "18"}),
        ("x264 fast      CRF18", "h264", {"preset": "fast", "crf": "18"}),
        ("x264 ultrafast CRF23", "h264", {"preset": "ultrafast", "crf": "23"}),
    ]

    # Check NVENC
    has_nvenc = False
    try:
        av.codec.Codec("h264_nvenc", "w")
        has_nvenc = True
    except Exception:
        pass

    if has_nvenc:
        configs.extend([
            ("NVENC p1 (fastest)",  "h264_nvenc", {"preset": "p1", "rc": "constqp", "qp": "18"}),
            ("NVENC p2",            "h264_nvenc", {"preset": "p2", "rc": "constqp", "qp": "18"}),
            ("NVENC p4 (balanced)", "h264_nvenc", {"preset": "p4", "rc": "constqp", "qp": "18"}),
            ("NVENC p7 (quality)",  "h264_nvenc", {"preset": "p7", "rc": "constqp", "qp": "18"}),
        ])

    results = []
    for label, codec, opts in configs:
        tmp = _tmpfile()
        gc.collect()

        t0 = time.perf_counter()
        with av.open(tmp, mode="w") as out:
            stream = out.add_stream(codec, rate=int(fps))
            stream.width = W2
            stream.height = H2
            stream.pix_fmt = "yuv420p"
            stream.options = opts
            _encode = stream.encode
            _mux = out.mux
            _from_ndarray = av.VideoFrame.from_ndarray
            for frame_np in frames_np[:, :H2, :W2, :]:
                for pkt in _encode(_from_ndarray(frame_np, format="rgb24")):
                    _mux(pkt)
            for pkt in _encode():
                _mux(pkt)
        dt = time.perf_counter() - t0

        sz = os.path.getsize(tmp) / 1024**2
        enc_fps = N / dt
        results.append((label, enc_fps, sz, dt))
        log.info(f"  {label:28s}  {enc_fps:6.0f} fps  {dt:5.2f}s  {sz:5.1f}MB")
        os.unlink(tmp)

    if results:
        best = max(results, key=lambda x: x[1])
        log.info(f"  → Best: {best[0]} ({best[1]:.0f} fps)")
    return results


def bench_encode_thread_count(frames_np: np.ndarray, fps: float):
    """Test encoder thread_count effect on libx264."""
    log.info(f"\n{'='*70}")
    log.info("ENCODE: libx264 thread_count effect")
    log.info(f"{'='*70}")

    N, H, W, C = frames_np.shape
    H2 = H - H % 2
    W2 = W - W % 2

    results = []
    for tc in [1, 2, 4, 8, 0]:  # 0 = auto
        tmp = _tmpfile()
        gc.collect()

        t0 = time.perf_counter()
        with av.open(tmp, mode="w") as out:
            stream = out.add_stream("h264", rate=int(fps))
            stream.width = W2
            stream.height = H2
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": "ultrafast", "crf": "18"}
            if tc > 0:
                stream.codec_context.thread_count = tc
                stream.codec_context.thread_type = av.codec.context.ThreadType.FRAME
            for frame_np in frames_np[:, :H2, :W2, :]:
                for pkt in stream.encode(av.VideoFrame.from_ndarray(frame_np, format="rgb24")):
                    out.mux(pkt)
            for pkt in stream.encode():
                out.mux(pkt)
        dt = time.perf_counter() - t0

        label = f"threads={tc}" if tc > 0 else "threads=auto"
        enc_fps = N / dt
        results.append((label, enc_fps, dt))
        log.info(f"  {label:14s}  {enc_fps:6.0f} fps  {dt:5.2f}s")
        os.unlink(tmp)

    return results


def bench_encode_from_ndarray_overhead(frames_np: np.ndarray, fps: float):
    """Measure VideoFrame.from_ndarray overhead vs total encode time."""
    log.info(f"\n{'='*70}")
    log.info("ENCODE: from_ndarray overhead breakdown")
    log.info(f"{'='*70}")

    N, H, W, C = frames_np.shape
    H2 = H - H % 2
    W2 = W - W % 2

    # Step 1: measure from_ndarray only (no encode)
    gc.collect()
    t0 = time.perf_counter()
    av_frames = []
    for frame_np in frames_np[:, :H2, :W2, :]:
        av_frames.append(av.VideoFrame.from_ndarray(frame_np, format="rgb24"))
    t_from_ndarray = time.perf_counter() - t0

    # Step 2: measure encode with pre-built frames
    tmp = _tmpfile()
    gc.collect()
    t0 = time.perf_counter()
    with av.open(tmp, mode="w") as out:
        stream = out.add_stream("h264", rate=int(fps))
        stream.width = W2
        stream.height = H2
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "18"}
        for f in av_frames:
            for pkt in stream.encode(f):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    t_encode_only = time.perf_counter() - t0
    os.unlink(tmp)

    # Step 3: measure combined (current approach)
    tmp = _tmpfile()
    gc.collect()
    t0 = time.perf_counter()
    with av.open(tmp, mode="w") as out:
        stream = out.add_stream("h264", rate=int(fps))
        stream.width = W2
        stream.height = H2
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast", "crf": "18"}
        for frame_np in frames_np[:, :H2, :W2, :]:
            f = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for pkt in stream.encode(f):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    t_combined = time.perf_counter() - t0
    os.unlink(tmp)

    del av_frames
    gc.collect()

    log.info(f"  from_ndarray only:  {t_from_ndarray:.3f}s  ({N/t_from_ndarray:.0f} fps)")
    log.info(f"  encode only:        {t_encode_only:.3f}s  ({N/t_encode_only:.0f} fps)")
    log.info(f"  combined:           {t_combined:.3f}s  ({N/t_combined:.0f} fps)")
    log.info(f"  → from_ndarray = {t_from_ndarray/t_combined*100:.0f}% of total encode time")



# ═════════════════════════════════════════════════════════════════════════════
# MUX overhead benchmark
# ═════════════════════════════════════════════════════════════════════════════

def bench_mux_overhead(path: str, max_frames: int):
    """Measure ffmpeg mux subprocess overhead vs PyAV stream copy."""
    log.info(f"\n{'='*70}")
    log.info("MUX: ffmpeg subprocess vs PyAV stream copy")
    log.info(f"{'='*70}")

    import subprocess

    # First create a video-only and audio-only file
    tmp_video = _tmpfile(".mp4")
    tmp_audio = _tmpfile(".wav")

    # Extract audio
    subprocess.run([
        "ffmpeg", "-y", "-i", path, "-vn", "-acodec", "pcm_s16le",
        "-map", "0:a:0", tmp_audio,
    ], capture_output=True)

    has_audio = os.path.exists(tmp_audio) and os.path.getsize(tmp_audio) > 100

    # Create a short video-only file
    subprocess.run([
        "ffmpeg", "-y", "-i", path, "-an", "-c:v", "copy",
        "-frames:v", str(max_frames), tmp_video,
    ], capture_output=True)

    if not has_audio:
        log.info("  No audio track — skipping mux benchmark")
        for f in [tmp_video, tmp_audio]:
            if os.path.exists(f):
                os.unlink(f)
        return

    video_mb = os.path.getsize(tmp_video) / 1024**2
    audio_mb = os.path.getsize(tmp_audio) / 1024**2
    log.info(f"  Video: {video_mb:.1f}MB, Audio: {audio_mb:.1f}MB")

    # Method 1: ffmpeg subprocess mux (current approach)
    tmp_out1 = _tmpfile(".mp4")
    t0 = time.perf_counter()
    subprocess.run([
        "ffmpeg", "-y",
        "-i", tmp_video, "-i", tmp_audio,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest",
        tmp_out1,
    ], capture_output=True)
    t_ffmpeg = time.perf_counter() - t0

    # Method 2: PyAV stream copy mux
    tmp_out2 = _tmpfile(".mp4")
    t0 = time.perf_counter()
    try:
        with av.open(tmp_out2, mode="w") as out:
            with av.open(tmp_video) as inv:
                with av.open(tmp_audio) as ina:
                    vs_in = inv.streams.video[0]
                    as_in = ina.streams.audio[0]
                    vs_out = out.add_stream(template=vs_in)
                    as_out = out.add_stream(template=as_in)
                    for pkt in inv.demux(vs_in):
                        if pkt.dts is None:
                            continue
                        pkt.stream = vs_out
                        out.mux(pkt)
                    for pkt in ina.demux(as_in):
                        if pkt.dts is None:
                            continue
                        pkt.stream = as_out
                        out.mux(pkt)
        t_pyav = time.perf_counter() - t0
        pyav_ok = True
    except Exception as e:
        t_pyav = time.perf_counter() - t0
        pyav_ok = False
        log.info(f"  PyAV mux failed: {e}")

    log.info(f"  ffmpeg subprocess:  {t_ffmpeg:.2f}s")
    if pyav_ok:
        log.info(f"  PyAV stream copy:   {t_pyav:.2f}s  "
                 f"({'faster' if t_pyav < t_ffmpeg else 'slower'} "
                 f"{abs(t_pyav-t_ffmpeg)/t_ffmpeg*100:.0f}%)")
    else:
        log.info(f"  PyAV stream copy:   FAILED")

    for f in [tmp_video, tmp_audio, tmp_out1, tmp_out2]:
        if os.path.exists(f):
            os.unlink(f)


# ═════════════════════════════════════════════════════════════════════════════
# Full pipeline timing (decode → embed-placeholder → encode)
# ═════════════════════════════════════════════════════════════════════════════

def bench_pipeline_io_ratio(path: str, max_frames: int):
    """Measure what % of total pipeline time is decode + encode (no GPU)."""
    log.info(f"\n{'='*70}")
    log.info("PIPELINE: I/O ratio (decode + encode vs total)")
    log.info(f"{'='*70}")

    # Decode
    t0 = time.perf_counter()
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "FRAME"
        fps = float(s.average_rate or s.guessed_rate or 25)
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    t_decode = time.perf_counter() - t0
    n = len(frames)
    H, W = frames[0].shape[0], frames[0].shape[1]

    # "Embed" placeholder — just torch operations (simulates CPU work)
    t0 = time.perf_counter()
    for i in range(0, n, 45):
        chunk = frames[i:i+45]
        batch = np.stack(chunk)
        t_batch = torch.from_numpy(batch).float().div_(255.0)
        # Simulate some work
        t_batch = t_batch * 0.95 + 0.05
        result = (t_batch.clamp_(0, 1).mul_(255)).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        del t_batch, batch
    t_process = time.perf_counter() - t0

    # Encode
    frames_np = np.stack(frames)
    H2 = H - H % 2
    W2 = W - W % 2

    # Test both ultrafast and current "fast"
    for preset in ["ultrafast", "fast"]:
        tmp = _tmpfile()
        t0 = time.perf_counter()
        with av.open(tmp, mode="w") as out:
            stream = out.add_stream("h264", rate=int(fps))
            stream.width = W2
            stream.height = H2
            stream.pix_fmt = "yuv420p"
            stream.options = {"preset": preset, "crf": "18"}
            for frame_np in frames_np[:, :H2, :W2, :]:
                f = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
                for pkt in stream.encode(f):
                    out.mux(pkt)
            for pkt in stream.encode():
                out.mux(pkt)
        t_encode = time.perf_counter() - t0
        sz = os.path.getsize(tmp) / 1024**2
        os.unlink(tmp)

        total = t_decode + t_process + t_encode
        log.info(f"\n  Preset: {preset}")
        log.info(f"    Decode:   {t_decode:.2f}s  ({t_decode/total*100:4.1f}%)  {n/t_decode:.0f} fps")
        log.info(f"    Process:  {t_process:.2f}s  ({t_process/total*100:4.1f}%)")
        log.info(f"    Encode:   {t_encode:.2f}s  ({t_encode/total*100:4.1f}%)  {n/t_encode:.0f} fps  → {sz:.1f}MB")
        log.info(f"    Total:    {total:.2f}s  ({n/total:.0f} fps)")
        log.info(f"    I/O ratio: {(t_decode+t_encode)/total*100:.0f}%")

    del frames, frames_np
    gc.collect()


# ═════════════════════════════════════════════════════════════════════════════
# Summary & recommendations
# ═════════════════════════════════════════════════════════════════════════════

def print_recommendations():
    log.info(f"\n{'='*70}")
    log.info("RECOMMENDATIONS")
    log.info(f"{'='*70}")
    log.info("""
  DECODE:
    1. thread_type="AUTO" thường nhanh hơn "FRAME" 10-20% cho H.264
    2. thread_count=4-8 tối ưu cho hầu hết CPU
    3. torch.as_tensor() thay vì from_numpy() tránh 1 copy
    4. div_(255.0) in-place thay vì / 255.0 tiết kiệm 1 allocation

  ENCODE:
    - Nếu có NVENC: dùng p1 hoặc p2, nhanh hơn libx264 5-10x
    - Nếu không NVENC: "ultrafast" nhanh hơn "fast" ~2x, chất lượng
      gần như không khác biệt ở CRF 18
    - Encoder thread_count > 1 giúp libx264 nhưng không ảnh hưởng NVENC
    - from_ndarray overhead ~10-15% encode time — khó tránh với PyAV

  MUX:
    - ffmpeg subprocess mux thêm 2-5s cho video 500MB
    - Có thể tránh bằng cách encode trực tiếp vào container có audio
    """)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Decode/Encode Benchmark")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--frames", type=int, default=300, help="Max frames to test")
    args = parser.parse_args()

    assert os.path.isfile(args.input), f"File not found: {args.input}"

    # Video info
    with av.open(args.input) as c:
        s = c.streams.video[0]
        log.info(f"Input: {args.input}")
        log.info(f"Resolution: {s.width}x{s.height}")
        log.info(f"Codec: {s.codec_context.name}")
        log.info(f"FPS: {float(s.average_rate or s.guessed_rate or 25):.1f}")
        log.info(f"Frames: {s.frames or '?'}")
        log.info(f"Testing with {args.frames} frames")

    # Run benchmarks
    bench_decode_thread_modes(args.input, args.frames)
    bench_decode_format_conversion(args.input, args.frames)
    bench_decode_batch_stack(args.input, args.frames)

    # Decode frames for encode benchmarks
    log.info("\nDecoding frames for encode benchmarks...")
    frames_np, fps = _decode_frames_np(args.input, args.frames)
    log.info(f"Decoded {frames_np.shape[0]} frames")

    bench_encode_presets(frames_np, fps)
    bench_encode_thread_count(frames_np, fps)
    bench_encode_from_ndarray_overhead(frames_np, fps)

    del frames_np
    gc.collect()

    bench_mux_overhead(args.input, args.frames)
    bench_pipeline_io_ratio(args.input, args.frames)
    print_recommendations()


if __name__ == "__main__":
    main()
