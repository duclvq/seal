"""
Test optimized inference pipeline: NVENC + GPU_BLEND + FP16.
Compares before/after speed and verifies watermark quality.

Usage:
    python test_optimizations_v2.py --input ../data/val/gi-diploma-66013582898105388892353.mp4
"""
import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "audioseal" / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "audioseal" / "src"))

os.environ["NO_TORCH_COMPILE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_embed_and_extract(input_path: str):
    """Test full embed → extract round-trip with optimized pipeline."""
    from worker import (
        load_video_model, embed_to_file,
        _NVENC_AVAILABLE, GPU_BLEND, USE_FP16, PIPELINE, FAST_EMBED,
    )
    from videoseal.utils.cfg import setup_model_from_checkpoint

    ckpt = str(_PROJECT_ROOT / "output" / "run2_video" / "checkpoint350.pth")

    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    log.info(f"\nConfig:")
    log.info(f"  FAST_EMBED: {FAST_EMBED}")
    log.info(f"  GPU_BLEND:  {GPU_BLEND}")
    log.info(f"  NVENC:      {_NVENC_AVAILABLE}")
    log.info(f"  FP16:       {USE_FP16}")
    log.info(f"  PIPELINE:   {PIPELINE}")

    # Load model
    log.info("\nLoading model...")
    video_model = load_video_model(ckpt, DEVICE)
    log.info("Model loaded.")

    # Embed
    fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    def progress(done, total):
        if total > 0:
            pct = done / total * 100
            print(f"\r  Progress: {done}/{total} ({pct:.0f}%)", end="", flush=True)

    log.info(f"\nEmbedding: {input_path}")
    t0 = time.perf_counter()
    result = embed_to_file(
        input_path=input_path,
        output_path=tmp_out,
        video_model=video_model,
        watermark_text="test",
        device=DEVICE,
        progress_callback=progress,
    )
    dt = time.perf_counter() - t0
    print()  # newline after progress

    fps_throughput = result["total_frames"] / dt if dt > 0 else 0
    log.info(f"\nResults:")
    log.info(f"  Mode:        {result['mode']}")
    log.info(f"  Frames:      {result['total_frames']}")
    log.info(f"  Resolution:  {result['resolution']}")
    log.info(f"  Time:        {dt:.1f}s")
    log.info(f"  Throughput:  {fps_throughput:.1f} fps")
    log.info(f"  OOM splits:  {result['oom_splits']}")

    if DEVICE.type == "cuda":
        alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
        peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
        log.info(f"  GPU alloc:   {alloc:.0f} MB")
        log.info(f"  GPU peak:    {peak:.0f} MB")

    # Verify watermark extraction
    log.info(f"\nVerifying watermark extraction...")
    extract_model = setup_model_from_checkpoint(ckpt)
    extract_model.eval().to(DEVICE)

    from web_demo.core.video_io import load_video_tensor
    from web_demo.core.ecc import msg_tensor_to_text_bch

    video, fps = load_video_tensor(tmp_out, max_frames=150)
    with torch.no_grad():
        msg = extract_model.extract_message(video.to(DEVICE), aggregation="squared_avg")

    decode = msg_tensor_to_text_bch(msg.cpu())
    log.info(f"  Extracted text: {decode['decoded_text']}")
    log.info(f"  Correctable:    {decode['correctable']}")
    log.info(f"  Bit errors:     {decode.get('bit_errors', 'N/A')}")

    # Verify output file size and bitrate
    out_size = os.path.getsize(tmp_out) / 1024 / 1024
    in_size = os.path.getsize(input_path) / 1024 / 1024
    from worker import _probe_bitrate
    in_br = _probe_bitrate(input_path)
    out_br = _probe_bitrate(tmp_out)
    log.info(f"\n  Input size:    {in_size:.1f} MB  ({in_br} kbps)")
    log.info(f"  Output size:   {out_size:.1f} MB  ({out_br} kbps)")
    log.info(f"  Size ratio:    {out_size/in_size:.2f}x")
    if in_br > 0 and out_br > 0:
        br_ratio = out_br / in_br
        log.info(f"  Bitrate ratio: {br_ratio:.2f}x (target ~1.0)")
        if 0.8 <= br_ratio <= 1.3:
            log.info(f"  ✓ Bitrate match OK")
        else:
            log.info(f"  ⚠ Bitrate mismatch (expected 0.8-1.3x)")

    os.unlink(tmp_out)

    # PASS/FAIL
    if decode["correctable"]:
        log.info(f"\n✓ PASS: Watermark correctly embedded and extracted")
    else:
        log.info(f"\n✗ FAIL: Watermark extraction failed")

    return {
        "time": dt,
        "fps": fps_throughput,
        "frames": result["total_frames"],
        "mode": result["mode"],
        "correctable": decode["correctable"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    test_embed_and_extract(args.input)
