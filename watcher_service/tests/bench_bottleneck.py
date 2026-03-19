"""
Bottleneck Profiler — Phân tích chi tiết từng giai đoạn pipeline.
Đo: RAM, VRAM, thời gian, throughput, và phát hiện memory leak.

Usage:
    python bench_bottleneck.py --input ../data/val/gi-diploma-66013582898105388892353.mp4
    python bench_bottleneck.py --input ../data/val/gi-diploma-66013582898105388892353.mp4 --chunk 30 45 60 90
"""
import argparse
import gc
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import psutil

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["NO_TORCH_COMPILE"] = "1"

import av
import numpy as np
import torch
from torch.nn import functional as F

from videoseal.utils.cfg import setup_model_from_checkpoint
from web_demo.core.ecc import text_to_msg_tensor_bch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INTERP = dict(mode="bilinear", align_corners=False, antialias=True)


# ═════════════════════════════════════════════════════════════════════════════
# Memory helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ram_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024**2


def _vram_mb(device=None) -> dict:
    """VRAM stats: allocated, reserved, peak."""
    if not torch.cuda.is_available():
        return {"alloc": 0, "reserved": 0, "peak": 0, "total": 0}
    d = device or DEVICE
    return {
        "alloc": torch.cuda.memory_allocated(d) / 1024**2,
        "reserved": torch.cuda.memory_reserved(d) / 1024**2,
        "peak": torch.cuda.max_memory_allocated(d) / 1024**2,
        "total": torch.cuda.get_device_properties(d).total_memory / 1024**2,
    }


def _reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()


class MemSnapshot:
    """Capture RAM + VRAM at a point in time."""
    def __init__(self, label=""):
        self.label = label
        self.ram = _ram_mb()
        self.vram = _vram_mb()
        self.time = time.perf_counter()

    def __repr__(self):
        v = self.vram
        return (f"[{self.label}] RAM={self.ram:.0f}MB  "
                f"VRAM alloc={v['alloc']:.0f}MB reserved={v['reserved']:.0f}MB "
                f"peak={v['peak']:.0f}MB")



# ═════════════════════════════════════════════════════════════════════════════
# Stage profilers
# ═════════════════════════════════════════════════════════════════════════════

def profile_decode(path: str, max_frames: int = 600) -> dict:
    """Profile video decoding: time, RAM growth, frame sizes."""
    gc.collect()
    snap_before = MemSnapshot("decode-before")

    frames = []
    t0 = time.perf_counter()
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "FRAME"
        w, h = stream.width, stream.height
        fps = float(stream.average_rate or stream.guessed_rate or 25)
        for frame in container.decode(stream):
            t = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
            frames.append(t)
            if len(frames) >= max_frames:
                break
    dt = time.perf_counter() - t0

    snap_after = MemSnapshot("decode-after")
    n = len(frames)

    # Per-frame memory: C*H*W * 1 byte (uint8)
    frame_bytes = 3 * h * w
    total_frames_mb = n * frame_bytes / 1024**2

    result = {
        "stage": "decode",
        "frames": n,
        "resolution": f"{w}x{h}",
        "fps_source": fps,
        "time_s": dt,
        "decode_fps": n / dt if dt > 0 else 0,
        "ram_before_mb": snap_before.ram,
        "ram_after_mb": snap_after.ram,
        "ram_growth_mb": snap_after.ram - snap_before.ram,
        "frames_in_ram_mb": total_frames_mb,
        "per_frame_kb": frame_bytes / 1024,
    }

    del frames
    gc.collect()
    return result


def profile_stack_normalize(frames_raw: list, chunk_size: int) -> dict:
    """Profile torch.stack + float/255 normalization."""
    gc.collect()
    snap_before = MemSnapshot("stack-before")

    chunk = frames_raw[:chunk_size]
    t0 = time.perf_counter()
    batch = torch.stack(chunk).float() / 255.0
    dt = time.perf_counter() - t0

    snap_after = MemSnapshot("stack-after")
    N, C, H, W = batch.shape

    # float32 tensor memory
    tensor_mb = N * C * H * W * 4 / 1024**2

    result = {
        "stage": "stack+normalize",
        "frames": N,
        "time_s": dt,
        "tensor_mb": tensor_mb,
        "ram_growth_mb": snap_after.ram - snap_before.ram,
    }

    del batch
    gc.collect()
    return result


def profile_resize_down(batch: torch.Tensor, img_size: int = 256) -> dict:
    """Profile CPU resize down to model input size."""
    N, C, H, W = batch.shape
    gc.collect()
    snap_before = MemSnapshot("resize-down-before")

    t0 = time.perf_counter()
    if H != img_size or W != img_size:
        batch_256 = F.interpolate(batch, size=(img_size, img_size), **_INTERP)
    else:
        batch_256 = batch
    dt = time.perf_counter() - t0

    snap_after = MemSnapshot("resize-down-after")
    tensor_mb = batch_256.numel() * 4 / 1024**2

    result = {
        "stage": "resize_down",
        "from": f"{W}x{H}",
        "to": f"{img_size}x{img_size}",
        "frames": N,
        "time_s": dt,
        "output_mb": tensor_mb,
        "ram_growth_mb": snap_after.ram - snap_before.ram,
    }

    del batch_256
    gc.collect()
    return result


def profile_gpu_inference(model, batch_256: torch.Tensor, msg_gpu: torch.Tensor,
                          use_fp16: bool = False) -> dict:
    """Profile GPU inference (embedder forward pass)."""
    _reset_vram_stats()
    snap_before = MemSnapshot("gpu-infer-before")

    step_size = model.step_size
    ck_size = model.chunk_size
    N = batch_256.shape[0]
    msgs = msg_gpu[:1].repeat(ck_size, 1)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    all_deltas = []
    for ii in range(0, len(batch_256[::step_size]), ck_size):
        nimgs = min(ck_size, len(batch_256[::step_size]) - ii)
        start = ii * step_size
        end = start + nimgs * step_size
        ck = batch_256[start:end].to(DEVICE)
        key = ck[::step_size]
        if model.embedder.yuv:
            key = model.rgb2yuv(key)[:, 0:1]
        ck_msgs = msgs[:nimgs] if nimgs < ck_size else msgs
        with torch.no_grad():
            if use_fp16:
                with torch.amp.autocast("cuda"):
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
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    snap_after = MemSnapshot("gpu-infer-after")

    result = {
        "stage": "gpu_inference",
        "frames": N,
        "time_s": dt,
        "fps": N / dt if dt > 0 else 0,
        "fp16": use_fp16,
        "vram_peak_mb": snap_after.vram["peak"],
        "vram_alloc_mb": snap_after.vram["alloc"],
        "ram_growth_mb": snap_after.ram - snap_before.ram,
        "delta_tensor_mb": delta_256.numel() * 4 / 1024**2,
    }

    del delta_256, all_deltas
    gc.collect()
    return result


def profile_resize_up_blend(batch: torch.Tensor, delta_256: torch.Tensor,
                            model, on_gpu: bool = False) -> dict:
    """Profile resize_up + blend (CPU or GPU)."""
    N, C, H, W = batch.shape
    img_size = model.img_size

    _reset_vram_stats()
    snap_before = MemSnapshot("blend-before")
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    if on_gpu:
        frames_d = batch.to(DEVICE)
        delta_d = delta_256.to(DEVICE)
        if H != img_size or W != img_size:
            delta_full = F.interpolate(delta_d, size=(H, W), **_INTERP)
        else:
            delta_full = delta_d
        result_t = float(model.blender.scaling_i) * frames_d + float(model.blender.scaling_w) * delta_full
        if model.clamp:
            result_t = torch.clamp(result_t, 0, 1)
        _ = result_t.cpu()
        del frames_d, delta_d, delta_full, result_t
    else:
        if H != img_size or W != img_size:
            delta_full = F.interpolate(delta_256, size=(H, W), **_INTERP)
        else:
            delta_full = delta_256
        result_t = float(model.blender.scaling_i) * batch + float(model.blender.scaling_w) * delta_full
        if model.clamp:
            result_t = torch.clamp(result_t, 0, 1)
        del delta_full, result_t

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.perf_counter() - t0

    snap_after = MemSnapshot("blend-after")

    # Estimate peak RAM for CPU blend: batch + delta_full + result = 3x full-res float32
    full_res_mb = N * C * H * W * 4 / 1024**2

    return {
        "stage": "resize_up+blend",
        "gpu": on_gpu,
        "frames": N,
        "time_s": dt,
        "fps": N / dt if dt > 0 else 0,
        "full_res_tensor_mb": full_res_mb,
        "peak_ram_estimate_mb": full_res_mb * 3 if not on_gpu else full_res_mb,
        "vram_peak_mb": snap_after.vram["peak"] if on_gpu else 0,
        "ram_growth_mb": snap_after.ram - snap_before.ram,
    }


def profile_to_numpy(batch_wm: torch.Tensor) -> dict:
    """Profile float32→uint8→numpy conversion."""
    snap_before = MemSnapshot("numpy-before")

    t0 = time.perf_counter()
    vid_np = (batch_wm.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    dt = time.perf_counter() - t0

    snap_after = MemSnapshot("numpy-after")
    np_mb = vid_np.nbytes / 1024**2

    result = {
        "stage": "to_numpy",
        "time_s": dt,
        "numpy_mb": np_mb,
        "ram_growth_mb": snap_after.ram - snap_before.ram,
    }

    del vid_np
    gc.collect()
    return result


def profile_encode(frames_np: np.ndarray, fps: float = 25.0,
                   codec: str = "h264", crf: str = "18") -> dict:
    """Profile H.264 encoding."""
    N, H, W, C = frames_np.shape
    H2 = H - H % 2
    W2 = W - W % 2

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    snap_before = MemSnapshot("encode-before")
    t0 = time.perf_counter()

    with av.open(tmp, mode="w") as out:
        stream = out.add_stream(codec, rate=int(fps))
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
    snap_after = MemSnapshot("encode-after")
    out_size = os.path.getsize(tmp) / 1024**2
    os.unlink(tmp)

    return {
        "stage": "h264_encode",
        "codec": codec,
        "frames": N,
        "time_s": dt,
        "fps": N / dt if dt > 0 else 0,
        "output_mb": out_size,
        "ram_growth_mb": snap_after.ram - snap_before.ram,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline memory estimator
# ═════════════════════════════════════════════════════════════════════════════

def estimate_memory_usage(width: int, height: int, chunk_size: int,
                          gpu_blend: bool = True, pipeline: bool = True) -> dict:
    """
    Estimate peak RAM and VRAM usage for a given resolution + chunk_size.
    Returns dict with estimates in MB.
    """
    C = 3
    # Per-frame sizes
    frame_uint8 = C * height * width          # 1 byte per element
    frame_f32 = C * height * width * 4        # 4 bytes per element
    frame_256_f32 = C * 256 * 256 * 4

    n = chunk_size

    # ── RAM estimate ──────────────────────────────────────────────────────
    # Decode buffer: chunk_size frames as uint8 tensors
    ram_decode_buf = n * frame_uint8

    # Stack + normalize: float32 batch
    ram_batch_f32 = n * frame_f32

    # Resize down: 256x256 float32
    ram_batch_256 = n * frame_256_f32

    # Delta at 256x256
    ram_delta_256 = n * frame_256_f32

    if gpu_blend:
        # CPU only holds: decode_buf + batch_f32 + batch_256 (briefly)
        # GPU does resize_up + blend
        ram_peak = ram_decode_buf + ram_batch_f32 + ram_batch_256
    else:
        # CPU blend: batch_f32 + delta_full + result = 3x full-res float32
        ram_delta_full = n * frame_f32
        ram_result = n * frame_f32
        ram_peak = ram_decode_buf + ram_batch_f32 + ram_delta_full + ram_result

    # Pipeline mode: up to 3 chunks in flight (decode_q=3, encode_q=4)
    if pipeline:
        # decode_q holds up to 3 batches as uint8 tensors
        ram_pipeline_queues = 3 * n * frame_uint8
        # encode_q holds up to 4 numpy arrays (uint8)
        ram_encode_queue = 4 * n * frame_uint8
        ram_peak += ram_pipeline_queues + ram_encode_queue

    # Numpy output (uint8)
    ram_numpy = n * frame_uint8

    ram_total = ram_peak + ram_numpy

    # ── VRAM estimate ─────────────────────────────────────────────────────
    # Model weights (approximate: ~200MB for VideoSeal + AudioSeal)
    vram_model = 200 * 1024**2

    # Inference: chunk at 256x256 on GPU
    vram_inference = n * frame_256_f32

    # Activations (rough: 2x input during forward pass)
    vram_activations = 2 * n * frame_256_f32

    if gpu_blend:
        # Full-res frames + delta on GPU
        vram_blend = 2 * n * frame_f32  # frames + delta_full
        vram_peak = vram_model + max(vram_inference + vram_activations, vram_blend)
    else:
        vram_peak = vram_model + vram_inference + vram_activations

    return {
        "resolution": f"{width}x{height}",
        "chunk_size": n,
        "gpu_blend": gpu_blend,
        "pipeline": pipeline,
        "ram_peak_mb": ram_total / 1024**2,
        "ram_decode_buf_mb": ram_decode_buf / 1024**2,
        "ram_batch_f32_mb": ram_batch_f32 / 1024**2,
        "ram_pipeline_extra_mb": (ram_pipeline_queues + ram_encode_queue) / 1024**2 if pipeline else 0,
        "vram_peak_mb": vram_peak / 1024**2,
        "vram_model_mb": vram_model / 1024**2,
        "vram_inference_mb": (vram_inference + vram_activations) / 1024**2,
        "vram_blend_mb": (2 * n * frame_f32) / 1024**2 if gpu_blend else 0,
    }


def find_safe_chunk_size(width: int, height: int, ram_limit_mb: float,
                         vram_limit_mb: float, gpu_blend: bool = True,
                         pipeline: bool = True) -> int:
    """Binary search for the largest chunk_size that fits within limits."""
    lo, hi = 1, 300
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        est = estimate_memory_usage(width, height, mid, gpu_blend, pipeline)
        if est["ram_peak_mb"] <= ram_limit_mb and est["vram_peak_mb"] <= vram_limit_mb:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


# ═════════════════════════════════════════════════════════════════════════════
# Full profiler
# ═════════════════════════════════════════════════════════════════════════════

def run_full_profile(input_path: str, model, msg_gpu: torch.Tensor,
                     chunk_sizes: list[int] = None) -> list[dict]:
    """Run full pipeline profile for each chunk_size. Returns list of results."""
    if chunk_sizes is None:
        chunk_sizes = [30, 45, 60, 90]

    # Decode all frames once
    log.info(f"\n{'='*70}")
    log.info("PHASE 1: Decode")
    log.info(f"{'='*70}")
    dec = profile_decode(input_path, max_frames=max(chunk_sizes) * 2)
    log.info(f"  {dec['frames']} frames ({dec['resolution']}) in {dec['time_s']:.2f}s "
             f"= {dec['decode_fps']:.0f} fps")
    log.info(f"  RAM growth: {dec['ram_growth_mb']:.0f}MB  "
             f"(frames in RAM: {dec['frames_in_ram_mb']:.0f}MB)")

    # Re-decode for profiling (need the actual frames)
    frames_raw = []
    with av.open(input_path) as c:
        s = c.streams.video[0]
        s.thread_type = "FRAME"
        for f in c.decode(s):
            frames_raw.append(torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1))
            if len(frames_raw) >= max(chunk_sizes) * 2:
                break

    all_results = []

    for cs in chunk_sizes:
        if cs > len(frames_raw):
            log.info(f"\nSkipping chunk_size={cs} (only {len(frames_raw)} frames available)")
            continue

        log.info(f"\n{'='*70}")
        log.info(f"PHASE 2-6: Pipeline profile (chunk_size={cs})")
        log.info(f"{'='*70}")

        chunk = frames_raw[:cs]
        _reset_vram_stats()

        # 2. Stack + normalize
        t0 = time.perf_counter()
        batch = torch.stack(chunk).float() / 255.0
        t_stack = time.perf_counter() - t0
        N, C, H, W = batch.shape

        # 3. Resize down
        r_down = profile_resize_down(batch, model.img_size)

        # Prepare 256x256 batch for GPU inference
        batch_256 = F.interpolate(batch, size=(model.img_size, model.img_size), **_INTERP)

        # 4. GPU inference (FP32 and FP16)
        r_gpu_fp32 = profile_gpu_inference(model, batch_256, msg_gpu, use_fp16=False)
        r_gpu_fp16 = profile_gpu_inference(model, batch_256, msg_gpu, use_fp16=True)

        # Get delta for blend profiling
        step_size = model.step_size
        ck_size = model.chunk_size
        msgs = msg_gpu[:1].repeat(ck_size, 1)
        all_deltas = []
        for ii in range(0, len(batch_256[::step_size]), ck_size):
            nimgs = min(ck_size, len(batch_256[::step_size]) - ii)
            start = ii * step_size
            end = start + nimgs * step_size
            ck = batch_256[start:end].to(DEVICE)
            key = ck[::step_size]
            if model.embedder.yuv:
                key = model.rgb2yuv(key)[:, 0:1]
            ck_msgs = msgs[:nimgs] if nimgs < ck_size else msgs
            with torch.no_grad():
                preds = model.embedder(key, ck_msgs)
            preds = model._apply_video_mode(preds, len(ck), step_size, model.video_mode)
            if model.attenuation is not None:
                model.attenuation.to(DEVICE)
                hmaps = model.attenuation.heatmaps(ck)
                preds = hmaps * preds
            all_deltas.append(preds.cpu())
            del ck, preds
        delta_256 = torch.cat(all_deltas, dim=0)[:N]

        # 5. Blend (CPU vs GPU)
        r_blend_cpu = profile_resize_up_blend(batch, delta_256, model, on_gpu=False)
        r_blend_gpu = profile_resize_up_blend(batch, delta_256, model, on_gpu=True)

        # 6. To numpy
        si = float(model.blender.scaling_i)
        sw = float(model.blender.scaling_w)
        delta_full = F.interpolate(delta_256, size=(H, W), **_INTERP)
        wm_batch = si * batch + sw * delta_full
        wm_batch = torch.clamp(wm_batch, 0, 1)
        r_numpy = profile_to_numpy(wm_batch)

        # 7. Encode
        vid_np = (wm_batch.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        r_encode = profile_encode(vid_np, fps=25.0)

        # Check NVENC
        try:
            av.codec.Codec("h264_nvenc", "w")
            r_encode_nvenc = profile_encode(vid_np, fps=25.0, codec="h264_nvenc")
        except Exception:
            r_encode_nvenc = None

        del batch, batch_256, delta_256, delta_full, wm_batch, vid_np
        gc.collect()

        # Memory estimates
        est = estimate_memory_usage(W, H, cs, gpu_blend=True, pipeline=True)

        # ── Print results ─────────────────────────────────────────────────
        total_cpu = (t_stack + r_down["time_s"] + r_gpu_fp32["time_s"]
                     + r_blend_cpu["time_s"] + r_numpy["time_s"] + r_encode["time_s"])
        total_gpu_blend = (t_stack + r_down["time_s"] + r_gpu_fp32["time_s"]
                           + r_blend_gpu["time_s"] + r_numpy["time_s"] + r_encode["time_s"])

        log.info(f"\n  chunk_size={cs}, resolution={W}x{H}")
        log.info(f"  ┌─────────────────────┬──────────┬──────────┬──────────────────┐")
        log.info(f"  │ Stage               │ Time (s) │ FPS      │ Memory           │")
        log.info(f"  ├─────────────────────┼──────────┼──────────┼──────────────────┤")
        log.info(f"  │ Stack+normalize     │ {t_stack:8.3f} │ {cs/t_stack:8.1f} │ {est['ram_batch_f32_mb']:6.0f} MB RAM   │")
        log.info(f"  │ Resize down (CPU)   │ {r_down['time_s']:8.3f} │ {cs/r_down['time_s']:8.1f} │                  │")
        log.info(f"  │ GPU infer (FP32)    │ {r_gpu_fp32['time_s']:8.3f} │ {r_gpu_fp32['fps']:8.1f} │ {r_gpu_fp32['vram_peak_mb']:6.0f} MB VRAM │")
        log.info(f"  │ GPU infer (FP16)    │ {r_gpu_fp16['time_s']:8.3f} │ {r_gpu_fp16['fps']:8.1f} │ {r_gpu_fp16['vram_peak_mb']:6.0f} MB VRAM │")
        log.info(f"  │ Blend (CPU)         │ {r_blend_cpu['time_s']:8.3f} │ {r_blend_cpu['fps']:8.1f} │ {r_blend_cpu['peak_ram_estimate_mb']:6.0f} MB RAM   │")
        log.info(f"  │ Blend (GPU)         │ {r_blend_gpu['time_s']:8.3f} │ {r_blend_gpu['fps']:8.1f} │ {r_blend_gpu['vram_peak_mb']:6.0f} MB VRAM │")
        log.info(f"  │ To numpy            │ {r_numpy['time_s']:8.3f} │ {cs/r_numpy['time_s']:8.1f} │ {r_numpy['numpy_mb']:6.0f} MB RAM   │")
        log.info(f"  │ H264 encode (CPU)   │ {r_encode['time_s']:8.3f} │ {r_encode['fps']:8.1f} │                  │")
        if r_encode_nvenc:
            log.info(f"  │ H264 encode (NVENC) │ {r_encode_nvenc['time_s']:8.3f} │ {r_encode_nvenc['fps']:8.1f} │                  │")
        log.info(f"  ├─────────────────────┼──────────┼──────────┼──────────────────┤")
        log.info(f"  │ TOTAL (CPU blend)   │ {total_cpu:8.2f} │ {cs/total_cpu:8.1f} │                  │")
        log.info(f"  │ TOTAL (GPU blend)   │ {total_gpu_blend:8.2f} │ {cs/total_gpu_blend:8.1f} │                  │")
        log.info(f"  └─────────────────────┴──────────┴──────────┴──────────────────┘")

        log.info(f"\n  Memory estimates (pipeline mode):")
        log.info(f"    RAM peak:  {est['ram_peak_mb']:.0f} MB")
        log.info(f"    VRAM peak: {est['vram_peak_mb']:.0f} MB")

        # Identify bottleneck
        stages = {
            "stack": t_stack,
            "resize_down": r_down["time_s"],
            "gpu_infer": r_gpu_fp32["time_s"],
            "blend_cpu": r_blend_cpu["time_s"],
            "blend_gpu": r_blend_gpu["time_s"],
            "to_numpy": r_numpy["time_s"],
            "encode": r_encode["time_s"],
        }
        bottleneck = max(stages, key=stages.get)
        log.info(f"\n  ⚠ BOTTLENECK: {bottleneck} ({stages[bottleneck]:.3f}s, "
                 f"{stages[bottleneck]/total_cpu*100:.0f}% of pipeline)")

        all_results.append({
            "chunk_size": cs,
            "resolution": f"{W}x{H}",
            "stages": stages,
            "total_cpu_blend": total_cpu,
            "total_gpu_blend": total_gpu_blend,
            "bottleneck": bottleneck,
            "memory_estimate": est,
            "gpu_fp32": r_gpu_fp32,
            "gpu_fp16": r_gpu_fp16,
            "encode": r_encode,
            "encode_nvenc": r_encode_nvenc,
        })

    del frames_raw
    gc.collect()
    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Recommendations engine
# ═════════════════════════════════════════════════════════════════════════════

def generate_recommendations(results: list[dict], vram_total_mb: float,
                             ram_total_mb: float) -> list[str]:
    """Generate actionable recommendations based on profiling results."""
    recs = []

    if not results:
        return ["Không có kết quả profiling để phân tích."]

    r = results[0]  # Use first chunk_size result as baseline
    stages = r["stages"]
    est = r["memory_estimate"]

    # ── 1. VRAM safety ────────────────────────────────────────────────────
    vram_usage_pct = est["vram_peak_mb"] / vram_total_mb * 100
    if vram_usage_pct > 85:
        safe_cs = find_safe_chunk_size(
            int(r["resolution"].split("x")[0]),
            int(r["resolution"].split("x")[1]),
            ram_limit_mb=ram_total_mb * 0.7,
            vram_limit_mb=vram_total_mb * 0.75,
        )
        recs.append(
            f"⚠ VRAM usage cao ({vram_usage_pct:.0f}%). "
            f"Giảm CHUNK_SIZE xuống {safe_cs} để tránh OOM crash. "
            f"Hoặc giảm GPU_MEMORY_FRACTION xuống 0.75."
        )

    # ── 2. RAM safety ─────────────────────────────────────────────────────
    ram_usage_pct = est["ram_peak_mb"] / ram_total_mb * 100
    if ram_usage_pct > 60:
        recs.append(
            f"⚠ RAM peak estimate {est['ram_peak_mb']:.0f}MB ({ram_usage_pct:.0f}% of {ram_total_mb:.0f}MB). "
            f"Pipeline mode giữ tới 7 chunks trong queue. "
            f"Giảm CHUNK_SIZE hoặc tắt PIPELINE=false."
        )

    # ── 3. Bottleneck-specific ────────────────────────────────────────────
    bn = r["bottleneck"]
    if bn == "blend_cpu":
        recs.append(
            f"Bottleneck là CPU blend ({stages['blend_cpu']:.2f}s). "
            f"Bật GPU_BLEND=true sẽ nhanh hơn ~{stages['blend_cpu']/stages['blend_gpu']:.0f}x "
            f"(+{est['vram_blend_mb']:.0f}MB VRAM)."
        )
    elif bn == "gpu_infer":
        fp32_t = r["gpu_fp32"]["time_s"]
        fp16_t = r["gpu_fp16"]["time_s"]
        if fp16_t < fp32_t * 0.85:
            recs.append(
                f"GPU inference là bottleneck ({fp32_t:.2f}s). "
                f"FP16 nhanh hơn {fp32_t/fp16_t:.1f}x ({fp16_t:.2f}s). "
                f"Bật USE_FP16=true."
            )
        else:
            recs.append(
                f"GPU inference là bottleneck ({fp32_t:.2f}s). "
                f"FP16 không cải thiện nhiều trên GPU này. "
                f"Tăng CHUNK_SIZE có thể giúp GPU utilization tốt hơn."
            )
    elif bn == "encode":
        enc = r["encode"]
        nvenc = r.get("encode_nvenc")
        if nvenc and nvenc["fps"] > enc["fps"] * 1.5:
            recs.append(
                f"H264 encode là bottleneck ({enc['time_s']:.2f}s, {enc['fps']:.0f}fps). "
                f"NVENC nhanh hơn {nvenc['fps']/enc['fps']:.1f}x ({nvenc['fps']:.0f}fps). "
                f"Bật USE_NVENC=true."
            )
        else:
            recs.append(
                f"H264 encode là bottleneck ({enc['time_s']:.2f}s). "
                f"Dùng PIPELINE=true để overlap encode với GPU inference."
            )
    elif bn == "stack":
        recs.append(
            f"Stack+normalize chiếm {stages['stack']:.2f}s. "
            f"Đây là overhead cố định, giảm CHUNK_SIZE sẽ giảm thời gian này."
        )

    # ── 4. Pipeline mode ──────────────────────────────────────────────────
    if len(results) >= 2:
        r1 = results[0]
        r2 = results[-1]
        if r2["total_gpu_blend"] / r2["chunk_size"] < r1["total_gpu_blend"] / r1["chunk_size"]:
            recs.append(
                f"Chunk lớn hơn ({r2['chunk_size']}) hiệu quả hơn "
                f"({r2['chunk_size']/r2['total_gpu_blend']:.1f} vs "
                f"{r1['chunk_size']/r1['total_gpu_blend']:.1f} fps/chunk). "
                f"Tăng CHUNK_SIZE nếu RAM/VRAM cho phép."
            )

    # ── 5. Queue depth warning ────────────────────────────────────────────
    if est.get("ram_pipeline_extra_mb", 0) > 500:
        recs.append(
            f"Pipeline queues chiếm {est['ram_pipeline_extra_mb']:.0f}MB RAM. "
            f"Giảm decode_q maxsize từ 3→2 và encode_q từ 4→2 trong worker.py "
            f"sẽ tiết kiệm ~{est['ram_pipeline_extra_mb']*0.5:.0f}MB."
        )

    return recs


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Bottleneck Profiler")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--chunk", type=int, nargs="+", default=[30, 45, 60, 90],
                        help="Chunk sizes to test")
    parser.add_argument("--ckpt", type=str,
                        default=str(_PROJECT_ROOT / "output" / "run2_video" / "checkpoint350.pth"))
    args = parser.parse_args()

    assert os.path.isfile(args.input), f"File not found: {args.input}"
    assert os.path.isfile(args.ckpt), f"Checkpoint not found: {args.ckpt}"

    # System info
    ram_total = psutil.virtual_memory().total / 1024**2
    log.info(f"System RAM: {ram_total:.0f} MB")
    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        log.info(f"VRAM: {vram_total:.0f} MB")
    else:
        vram_total = 0

    # Load model
    log.info("\nLoading model...")
    model = setup_model_from_checkpoint(args.ckpt)
    model.eval().to(DEVICE)
    log.info(f"Model loaded. VRAM after load: {_vram_mb()['alloc']:.0f} MB")
    log.info(f"Model config: img_size={model.img_size}, step_size={model.step_size}, "
             f"chunk_size={model.chunk_size}")

    msg_t, _, _ = text_to_msg_tensor_bch("bn0", msg_bits=256)
    msg_gpu = msg_t.to(DEVICE)

    # Run profiling
    results = run_full_profile(args.input, model, msg_gpu, chunk_sizes=args.chunk)

    # Recommendations
    log.info(f"\n{'='*70}")
    log.info("RECOMMENDATIONS")
    log.info(f"{'='*70}")
    recs = generate_recommendations(results, vram_total, ram_total)
    for i, rec in enumerate(recs, 1):
        log.info(f"  {i}. {rec}")

    # Safe chunk size
    if DEVICE.type == "cuda":
        # Get resolution from first result
        if results:
            w, h = map(int, results[0]["resolution"].split("x"))
            safe = find_safe_chunk_size(w, h, ram_total * 0.7, vram_total * 0.75)
            log.info(f"\n  Recommended CHUNK_SIZE for {w}x{h}: {safe}")
            log.info(f"  (RAM limit: {ram_total*0.7:.0f}MB, VRAM limit: {vram_total*0.75:.0f}MB)")

    del msg_gpu, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
