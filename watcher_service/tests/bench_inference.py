"""
Benchmark inference pipeline bottlenecks on Windows.
Tests: decode, GPU embed, encode, and full pipeline with different configs.

Usage:
    python bench_inference.py --input ../data/val/gi-diploma-66013582898105388892353.mp4
"""
import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import av
import numpy as np
import torch
from torch.nn import functional as F

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PROJECT_ROOT / "audioseal" / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "audioseal" / "src"))

os.environ["NO_TORCH_COMPILE"] = "1"

from videoseal.utils.cfg import setup_model_from_checkpoint
from web_demo.core.ecc import text_to_msg_tensor_bch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CKPT = os.getenv("CKPT_PATH", str(_PROJECT_ROOT / "output" / "run2_video" / "checkpoint350.pth"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INTERP = dict(mode="bilinear", align_corners=False, antialias=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_model():
    model = setup_model_from_checkpoint(CKPT)
    model.eval().to(DEVICE)
    return model


def decode_frames_av(path: str, max_frames: int = 300):
    """Decode video frames using PyAV. Returns list of [C,H,W] uint8 tensors."""
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"
        for frame in container.decode(stream):
            t = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            frames.append(t)
            if len(frames) >= max_frames:
                break
    return frames


def bench_decode(path: str, max_frames: int = 300, runs: int = 3):
    """Benchmark: video decoding speed."""
    log.info(f"\n{'='*60}")
    log.info(f"BENCHMARK: Video Decode (PyAV)")
    log.info(f"{'='*60}")

    times = []
    n_frames = 0
    for r in range(runs):
        t0 = time.perf_counter()
        frames = decode_frames_av(path, max_frames)
        dt = time.perf_counter() - t0
        n_frames = len(frames)
        times.append(dt)
        h, w = frames[0].shape[1], frames[0].shape[2]
        del frames
        log.info(f"  Run {r+1}: {n_frames} frames ({w}x{h}) in {dt:.2f}s = {n_frames/dt:.1f} fps")

    avg = sum(times) / len(times)
    log.info(f"  AVG: {avg:.2f}s ({n_frames/avg:.1f} fps)")
    return avg, n_frames


def bench_gpu_embed(model, frames_batch: torch.Tensor, msg_gpu: torch.Tensor,
                    chunk_size: int = 30, use_fp16: bool = False):
    """Benchmark: GPU embedding only (no decode/encode)."""
    N, C, H, W = frames_batch.shape
    img_size = model.img_size

    # Resize to 256x256 on CPU
    if H != img_size or W != img_size:
        frames_256 = F.interpolate(frames_batch, size=(img_size, img_size), **_INTERP)
    else:
        frames_256 = frames_batch

    step_size = model.step_size
    ck_size = model.chunk_size
    msgs = msg_gpu[:1].repeat(ck_size, 1)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    for ii in range(0, len(frames_256[::step_size]), ck_size):
        nimgs = min(ck_size, len(frames_256[::step_size]) - ii)
        start = ii * step_size
        end = start + nimgs * step_size
        ck = frames_256[start:end].to(DEVICE)
        key = ck[::step_size]

        if model.embedder.yuv:
            key = model.rgb2yuv(key)[:, 0:1]

        ck_msgs = msgs[:nimgs] if nimgs < ck_size else msgs

        with torch.no_grad():
            if use_fp16:
                with torch.cuda.amp.autocast():
                    preds = model.embedder(key, ck_msgs)
            else:
                preds = model.embedder(key, ck_msgs)

        preds = model._apply_video_mode(preds, len(ck), step_size, model.video_mode)

        if model.attenuation is not None:
            model.attenuation.to(DEVICE)
            hmaps = model.attenuation.heatmaps(ck)
            preds = hmaps * preds

        _ = preds.cpu()
        del ck, preds

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    return dt


def bench_encode_h264(frames_np: np.ndarray, fps: float = 25.0, crf: str = "18"):
    """Benchmark: H.264 encoding via PyAV."""
    N, H, W, C = frames_np.shape
    H2 = H - H % 2
    W2 = W - W % 2

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    t0 = time.perf_counter()
    with av.open(tmp, mode="w") as out:
        stream = out.add_stream("h264", rate=int(fps))
        stream.width = W2
        stream.height = H2
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": crf, "preset": "fast"}

        for frame_np in frames_np[:, :H2, :W2, :]:
            f = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for pkt in stream.encode(f):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)

    dt = time.perf_counter() - t0
    os.unlink(tmp)
    return dt


def bench_blend(frames: torch.Tensor, delta_256: torch.Tensor, model,
                on_gpu: bool = False):
    """Benchmark: resize_up + blend step."""
    N, C, H, W = frames.shape
    img_size = model.img_size

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    if on_gpu:
        frames_d = frames.to(DEVICE)
        delta_d = delta_256.to(DEVICE)
        if H != img_size or W != img_size:
            delta_full = F.interpolate(delta_d, size=(H, W), **_INTERP)
        else:
            delta_full = delta_d
        result = float(model.blender.scaling_i) * frames_d + float(model.blender.scaling_w) * delta_full
        if model.clamp:
            result = torch.clamp(result, 0, 1)
        _ = result.cpu()
        del frames_d, delta_d, delta_full, result
    else:
        if H != img_size or W != img_size:
            delta_full = F.interpolate(delta_256, size=(H, W), **_INTERP)
        else:
            delta_full = delta_256
        result = float(model.blender.scaling_i) * frames + float(model.blender.scaling_w) * delta_full
        if model.clamp:
            result = torch.clamp(result, 0, 1)
        del delta_full, result

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    dt = time.perf_counter() - t0
    return dt


# ── Full pipeline benchmark ──────────────────────────────────────────────────

def bench_full_pipeline(model, path: str, msg_gpu: torch.Tensor,
                        chunk_size: int = 30, use_fast: bool = True,
                        gpu_blend: bool = False, use_fp16: bool = False):
    """Benchmark the full embed pipeline with timing breakdown."""
    log.info(f"\n  Config: chunk={chunk_size}, fast={use_fast}, gpu_blend={gpu_blend}, fp16={use_fp16}")

    img_size = model.img_size
    step_size = model.step_size
    ck_size = model.chunk_size
    msgs = msg_gpu[:1].repeat(ck_size, 1)

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    t_decode = 0
    t_gpu = 0
    t_blend = 0
    t_encode = 0
    total_frames = 0
    out_stream = None
    h_out = w_out = None

    t_total_start = time.perf_counter()

    with av.open(tmp, mode="w") as out_container:
        with av.open(path) as in_container:
            in_stream = in_container.streams.video[0]
            in_stream.thread_type = "FRAME"
            rate = in_stream.average_rate or in_stream.guessed_rate
            fps = float(rate) if rate else 25.0

            chunk = []
            for frame in in_container.decode(in_stream):
                td0 = time.perf_counter()
                t = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                chunk.append(t)
                t_decode += time.perf_counter() - td0

                if len(chunk) >= chunk_size:
                    frames_batch = torch.stack(chunk).float() / 255.0
                    chunk = []
                    N, C, H, W = frames_batch.shape

                    # GPU embed
                    tg0 = time.perf_counter()
                    if H != img_size or W != img_size:
                        frames_256 = F.interpolate(frames_batch, size=(img_size, img_size), **_INTERP)
                    else:
                        frames_256 = frames_batch

                    # Get delta
                    all_deltas = []
                    for ii in range(0, len(frames_256[::step_size]), ck_size):
                        nimgs = min(ck_size, len(frames_256[::step_size]) - ii)
                        start = ii * step_size
                        end = start + nimgs * step_size
                        ck = frames_256[start:end].to(DEVICE)
                        key = ck[::step_size]
                        if model.embedder.yuv:
                            key = model.rgb2yuv(key)[:, 0:1]
                        ck_msgs = msgs[:nimgs] if nimgs < ck_size else msgs
                        with torch.no_grad():
                            if use_fp16:
                                with torch.cuda.amp.autocast():
                                    preds = model.embedder(key, ck_msgs)
                            else:
                                preds = model.embedder(key, ck_msgs)
                        preds = model._apply_video_mode(preds, len(ck), step_size, model.video_mode)
                        if model.attenuation is not None:
                            model.attenuation.to(DEVICE)
                            hmaps = model.attenuation.heatmaps(ck)
                            preds = hmaps * preds
                        all_deltas.append(preds.cpu())
                        del ck, preds
                    delta_256 = torch.cat(all_deltas, dim=0)[:N]

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t_gpu += time.perf_counter() - tg0

                    # Blend
                    tb0 = time.perf_counter()
                    if gpu_blend:
                        frames_d = frames_batch.to(DEVICE)
                        delta_d = delta_256.to(DEVICE)
                        if H != img_size or W != img_size:
                            delta_full = F.interpolate(delta_d, size=(H, W), **_INTERP)
                        else:
                            delta_full = delta_d
                        result = float(model.blender.scaling_i) * frames_d + float(model.blender.scaling_w) * delta_full
                        if model.clamp:
                            result = torch.clamp(result, 0, 1)
                        wm_frames = result.cpu()
                        del frames_d, delta_d, delta_full, result
                    else:
                        if H != img_size or W != img_size:
                            delta_full = F.interpolate(delta_256, size=(H, W), **_INTERP)
                        else:
                            delta_full = delta_256
                        wm_frames = float(model.blender.scaling_i) * frames_batch + float(model.blender.scaling_w) * delta_full
                        if model.clamp:
                            wm_frames = torch.clamp(wm_frames, 0, 1)
                        del delta_full

                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    t_blend += time.perf_counter() - tb0

                    del frames_batch, delta_256

                    # Encode
                    te0 = time.perf_counter()
                    vid_np = (wm_frames.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                    del wm_frames

                    if out_stream is None:
                        h_out = vid_np.shape[1] - vid_np.shape[1] % 2
                        w_out = vid_np.shape[2] - vid_np.shape[2] % 2
                        out_stream = out_container.add_stream("h264", rate=int(fps))
                        out_stream.width = w_out
                        out_stream.height = h_out
                        out_stream.pix_fmt = "yuv420p"
                        out_stream.options = {"crf": "18", "preset": "fast"}

                    for fnp in vid_np[:, :h_out, :w_out, :]:
                        f = av.VideoFrame.from_ndarray(fnp, format="rgb24")
                        for pkt in out_stream.encode(f):
                            out_container.mux(pkt)
                    t_encode += time.perf_counter() - te0

                    total_frames += N

            # Flush remaining
            if chunk:
                frames_batch = torch.stack(chunk).float() / 255.0
                N = len(chunk)
                # simplified: just count time
                tg0 = time.perf_counter()
                if use_fast:
                    from worker import _embed_batch_fast
                    wm = _embed_batch_fast(frames_batch, model, msg_gpu, DEVICE)
                else:
                    from worker import _embed_batch_standard
                    wm = _embed_batch_standard(frames_batch, model, msg_gpu, DEVICE)
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                t_gpu += time.perf_counter() - tg0

                te0 = time.perf_counter()
                vid_np = (wm.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
                if out_stream is not None:
                    for fnp in vid_np[:, :h_out, :w_out, :]:
                        f = av.VideoFrame.from_ndarray(fnp, format="rgb24")
                        for pkt in out_stream.encode(f):
                            out_container.mux(pkt)
                t_encode += time.perf_counter() - te0
                total_frames += N

            if out_stream:
                for pkt in out_stream.encode():
                    out_container.mux(pkt)

    t_total = time.perf_counter() - t_total_start
    os.unlink(tmp)

    log.info(f"  Total: {total_frames} frames in {t_total:.2f}s = {total_frames/t_total:.1f} fps")
    log.info(f"    Decode:  {t_decode:.2f}s ({t_decode/t_total*100:.1f}%)")
    log.info(f"    GPU:     {t_gpu:.2f}s ({t_gpu/t_total*100:.1f}%)")
    log.info(f"    Blend:   {t_blend:.2f}s ({t_blend/t_total*100:.1f}%)")
    log.info(f"    Encode:  {t_encode:.2f}s ({t_encode/t_total*100:.1f}%)")
    log.info(f"    Other:   {t_total - t_decode - t_gpu - t_blend - t_encode:.2f}s")

    return {
        "total_frames": total_frames,
        "total_time": t_total,
        "fps": total_frames / t_total,
        "decode": t_decode,
        "gpu": t_gpu,
        "blend": t_blend,
        "encode": t_encode,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark inference bottlenecks")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--chunk_size", type=int, default=45)
    args = parser.parse_args()

    assert os.path.isfile(args.input), f"File not found: {args.input}"

    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load model
    log.info("\nLoading model...")
    model = load_model()
    log.info("Model loaded.")

    # Prepare message
    msg_tensor, _, _ = text_to_msg_tensor_bch("bn0", msg_bits=256)
    msg_gpu = msg_tensor.to(DEVICE)

    # ── 1. Decode benchmark ──────────────────────────────────────────────
    bench_decode(args.input, max_frames=args.max_frames)

    # ── 2. GPU embed benchmark (fp32 vs fp16) ────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"BENCHMARK: GPU Embed (fp32 vs fp16)")
    log.info(f"{'='*60}")

    frames_raw = decode_frames_av(args.input, max_frames=min(args.chunk_size, args.max_frames))
    frames_batch = torch.stack(frames_raw).float() / 255.0
    N, C, H, W = frames_batch.shape
    log.info(f"  Frames: {N} x {W}x{H}")

    # Warmup
    _ = bench_gpu_embed(model, frames_batch, msg_gpu, use_fp16=False)

    for fp16 in [False, True]:
        times = []
        for r in range(3):
            dt = bench_gpu_embed(model, frames_batch, msg_gpu, use_fp16=fp16)
            times.append(dt)
        avg = sum(times) / len(times)
        label = "FP16" if fp16 else "FP32"
        log.info(f"  {label}: {avg:.3f}s ({N/avg:.1f} fps) [{', '.join(f'{t:.3f}' for t in times)}]")

    # ── 3. Blend benchmark (CPU vs GPU) ──────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"BENCHMARK: Blend (CPU vs GPU)")
    log.info(f"{'='*60}")

    img_size = model.img_size
    frames_256 = F.interpolate(frames_batch, size=(img_size, img_size), **_INTERP)
    # Get a dummy delta
    dummy_delta = torch.randn_like(frames_256) * 0.01

    for on_gpu in [False, True]:
        times = []
        for r in range(3):
            dt = bench_blend(frames_batch, dummy_delta, model, on_gpu=on_gpu)
            times.append(dt)
        avg = sum(times) / len(times)
        label = "GPU" if on_gpu else "CPU"
        log.info(f"  {label}: {avg:.3f}s ({N/avg:.1f} fps) [{', '.join(f'{t:.3f}' for t in times)}]")

    del dummy_delta

    # ── 4. Encode benchmark ──────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"BENCHMARK: H.264 Encode (PyAV)")
    log.info(f"{'='*60}")

    frames_np = (frames_batch.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    for crf in ["18", "23", "28"]:
        times = []
        for r in range(3):
            dt = bench_encode_h264(frames_np, crf=crf)
            times.append(dt)
        avg = sum(times) / len(times)
        log.info(f"  CRF={crf}: {avg:.3f}s ({N/avg:.1f} fps) [{', '.join(f'{t:.3f}' for t in times)}]")

    del frames_np

    # ── 5. Full pipeline comparison ──────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"BENCHMARK: Full Pipeline Comparison")
    log.info(f"{'='*60}")

    configs = [
        {"use_fast": True,  "gpu_blend": False, "use_fp16": False},  # baseline
        {"use_fast": True,  "gpu_blend": True,  "use_fp16": False},  # + GPU blend
        {"use_fast": True,  "gpu_blend": False, "use_fp16": True},   # + FP16
        {"use_fast": True,  "gpu_blend": True,  "use_fp16": True},   # + both
    ]

    results = []
    for cfg in configs:
        r = bench_full_pipeline(model, args.input, msg_gpu,
                                chunk_size=args.chunk_size, **cfg)
        results.append((cfg, r))

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"SUMMARY")
    log.info(f"{'='*60}")
    for cfg, r in results:
        label = f"fast={cfg['use_fast']}, gpu_blend={cfg['gpu_blend']}, fp16={cfg['use_fp16']}"
        log.info(f"  [{label}] → {r['fps']:.1f} fps ({r['total_time']:.1f}s)")

    del frames_batch, msg_gpu


if __name__ == "__main__":
    main()
