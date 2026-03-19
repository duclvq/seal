"""
Quick throughput benchmark: embed N frames and measure steady-state fps.
Usage: python bench_throughput.py --input ../data/val/gi-diploma-66013582898105388892353.mp4 --frames 3000
"""
import argparse, logging, os, sys, tempfile, time
from pathlib import Path

import torch

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["NO_TORCH_COMPILE"] = "1"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(input_path: str, max_frames: int):
    from worker import (
        load_video_model, embed_to_file,
        _NVENC_AVAILABLE, GPU_BLEND, USE_FP16, PIPELINE, FAST_EMBED, CHUNK_SIZE,
    )

    ckpt = str(_PROJECT_ROOT / "output" / "run2_video" / "checkpoint350.pth")
    log.info(f"GPU: {torch.cuda.get_device_name()}")
    log.info(f"Config: CHUNK={CHUNK_SIZE} FAST={FAST_EMBED} GPU_BLEND={GPU_BLEND} "
             f"NVENC={_NVENC_AVAILABLE} FP16={USE_FP16} PIPE={PIPELINE}")

    video_model = load_video_model(ckpt, DEVICE)

    # Create a truncated copy with max_frames
    import av
    fd, tmp_in = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    fd2, tmp_out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd2)

    log.info(f"Preparing {max_frames} frames from {input_path}...")
    import subprocess
    # Use ffmpeg to trim video (avoids PyAV muxing issues)
    duration = max_frames / 24.0  # approximate
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-t", str(duration), "-c", "copy", tmp_in,
    ], capture_output=True)
    # Count actual frames
    with av.open(tmp_in) as inp:
        vs = inp.streams.video[0]
        count = vs.frames or max_frames
    log.info(f"Prepared ~{count} frames ({duration:.0f}s).")

    # Warmup
    log.info("Warmup run...")
    fd3, tmp_warm = tempfile.mkstemp(suffix=".mp4")
    os.close(fd3)
    embed_to_file(tmp_in, tmp_warm, video_model, "test", DEVICE)
    os.unlink(tmp_warm)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Timed run
    log.info("Timed run...")
    t0 = time.perf_counter()
    result = embed_to_file(tmp_in, tmp_out, video_model, "test", DEVICE)
    dt = time.perf_counter() - t0

    fps_out = result["total_frames"] / dt
    log.info(f"\n{'='*50}")
    log.info(f"RESULT: {result['total_frames']} frames in {dt:.1f}s = {fps_out:.1f} fps")
    log.info(f"Mode: {result['mode']}")
    if DEVICE.type == "cuda":
        peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        log.info(f"GPU peak: {peak:.0f} MB")

    # File size comparison
    in_sz = os.path.getsize(tmp_in) / 1024 / 1024
    out_sz = os.path.getsize(tmp_out) / 1024 / 1024
    log.info(f"Input: {in_sz:.1f} MB, Output: {out_sz:.1f} MB (ratio: {out_sz/in_sz:.2f}x)")
    log.info(f"{'='*50}")

    os.unlink(tmp_in)
    os.unlink(tmp_out)
    return fps_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--frames", type=int, default=3000)
    args = p.parse_args()
    run(args.input, args.frames)
