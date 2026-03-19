"""
Video + audio watermark embedding worker.
Uses: custom VideoSeal model, BCH-256 ECC, center_mask=True, AudioSeal for audio.

Two embed modes:
  - Standard (FAST_EMBED=false): sends full-res frames to GPU (original pipeline)
  - Fast     (FAST_EMBED=true):  resize on CPU → only 256×256 to GPU → blend on CPU
    Cuts GPU transfer by ~70× for 2K video. Same watermark quality.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import av
import torch
from torch.nn import functional as F

# ── Project root on sys.path ─────────────────────────────────────────────────
_SERVICE_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Make audioseal importable
_AUDIOSEAL_SRC = _PROJECT_ROOT / "audioseal" / "src"
if str(_AUDIOSEAL_SRC) not in sys.path:
    sys.path.insert(0, str(_AUDIOSEAL_SRC))

os.environ.setdefault("NO_TORCH_COMPILE", "1")

from videoseal.utils.cfg import setup_model_from_checkpoint
from web_demo.core.ecc      import text_to_msg_tensor_bch
from web_demo.core.video_io import (
    extract_audio_track,
    _mux_video_audio,
    _EXT_TO_FORMAT,
    _FFMPEG_CONVERT_EXT,
    _to_mp4_via_ffmpeg,
)

MSG_BITS    = 256
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE", "30"))
FAST_EMBED  = os.getenv("FAST_EMBED", "true").lower() == "true"
GPU_BLEND   = os.getenv("GPU_BLEND", "true").lower() == "true"   # GPU blend (2x faster)
PIPELINE    = os.getenv("PIPELINE", "true").lower() == "true"    # 3-stage parallel pipeline
USE_NVENC   = os.getenv("USE_NVENC", "true").lower() == "true"   # NVENC hardware encoder (6x faster)
USE_FP16    = os.getenv("USE_FP16", "false").lower() == "true"   # FP16 inference
NVENC_PRESET = os.getenv("NVENC_PRESET", "p4")                   # p1=fastest ... p7=slowest
DECODE_THREADS = int(os.getenv("DECODE_THREADS", "0"))            # 0=auto (min(8, cpu_count))
ENCODE_PRESET  = os.getenv("ENCODE_PRESET", "")                   # override libx264 preset (ultrafast/fast/etc)
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "1"))                 # parallel video processing
_oom_splits = 0  # counter for OOM auto-splits (per-video, reset in embed_to_file)
CRF        = "18"   # near-visually-lossless H.264

# ── Auto chunk size: reduce if VRAM is limited ───────────────────────────
def _auto_chunk_size(base_chunk: int, device: torch.device) -> int:
    """Reduce chunk_size if available VRAM is low. Prevents OOM crashes."""
    if device.type != "cuda":
        return base_chunk
    try:
        total = torch.cuda.get_device_properties(device).total_memory
        reserved = torch.cuda.memory_reserved(device)
        free = total - reserved
        free_mb = free / 1024**2

        # Heuristic: each frame at 1080p needs ~24MB VRAM (256x256 inference + blend)
        # With GPU_BLEND, full-res frames also go to GPU: ~12MB/frame for 1080p float32
        per_frame_mb = 36 if GPU_BLEND else 24
        max_safe_frames = int(free_mb * 0.7 / per_frame_mb)  # use 70% of free VRAM
        max_safe_frames = max(4, max_safe_frames)  # minimum 4 frames

        if max_safe_frames < base_chunk:
            return max_safe_frames
    except Exception:
        pass
    return base_chunk

# ── Thread safety for multi-worker GPU access ────────────────────────────
import threading as _threading
_cuda_lock = _threading.Lock()  # serialize CUDA model inference across workers

# ── NVENC availability check ─────────────────────────────────────────────────
_NVENC_AVAILABLE = False
if USE_NVENC:
    try:
        _test_codec = av.codec.Codec("h264_nvenc", "w")
        _NVENC_AVAILABLE = True
        del _test_codec
    except Exception:
        _NVENC_AVAILABLE = False

import numpy as np

# ── BT.709 RGB→YUV conversion matrices (GPU) ────────────────────────────
# Standard BT.709 for HD content:
#   Y  =  0.2126*R + 0.7152*G + 0.0722*B
#   Cb = -0.1146*R - 0.3854*G + 0.5000*B + 128
#   Cr =  0.5000*R - 0.4542*G - 0.0458*B + 128
_BT709_Y  = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)
_BT709_CB = torch.tensor([-0.1146, -0.3854, 0.5000], dtype=torch.float32)
_BT709_CR = torch.tensor([0.5000, -0.4542, -0.0458], dtype=torch.float32)

# Inverse BT.709 YUV→RGB matrix (for GPU decode)
# R = Y + 1.5748 * Cr'
# G = Y - 0.1873 * Cb' - 0.4681 * Cr'
# B = Y + 1.8556 * Cb'
# where Cb' = Cb - 128, Cr' = Cr - 128
_BT709_INV = torch.tensor([
    [1.0,  0.0,      1.5748],   # R
    [1.0, -0.1873,  -0.4681],   # G
    [1.0,  1.8556,   0.0   ],   # B
], dtype=torch.float32)  # shape [3, 3]


def _rgb_to_yuv420p_gpu(
    rgb: torch.Tensor,
) -> list[np.ndarray]:
    """
    Convert RGB uint8 tensor [N, H, W, 3] on GPU → list of YUV420P numpy arrays.
    Convenience wrapper: GPU compute + CPU transfer in one call.
    """
    packed = _rgb_to_yuv420p_gpu_compute(rgb)
    return _yuv420p_gpu_to_cpu(packed)


def _rgb_to_yuv420p_gpu_compute(
    rgb: torch.Tensor,
) -> torch.Tensor:
    """
    RGB uint8 [N, H, W, 3] on GPU → YUV420P packed tensor [N, H*3//2, W] uint8 on GPU.
    All heavy math (matmul, chroma subsampling) runs on GPU. No CPU transfer.
    """
    device = rgb.device
    N, H, W, _ = rgb.shape

    H2 = H - H % 2
    W2 = W - W % 2
    if H2 != H or W2 != W:
        rgb = rgb[:, :H2, :W2, :]
        H, W = H2, W2

    global _BT709_Y, _BT709_CB, _BT709_CR
    if _BT709_Y.device != device:
        _BT709_Y = _BT709_Y.to(device)
        _BT709_CB = _BT709_CB.to(device)
        _BT709_CR = _BT709_CR.to(device)

    rgb_f = rgb.float()
    Y = (rgb_f @ _BT709_Y).clamp_(0, 255).to(torch.uint8)
    rgb_sub = rgb_f.reshape(N, H // 2, 2, W // 2, 2, 3).mean(dim=(2, 4))
    Cb = (rgb_sub @ _BT709_CB + 128).clamp_(0, 255).to(torch.uint8)
    Cr = (rgb_sub @ _BT709_CR + 128).clamp_(0, 255).to(torch.uint8)
    del rgb_f, rgb_sub

    hh = H // 2
    hw = W // 2
    packed = torch.empty((N, H + hh, W), dtype=torch.uint8, device=device)
    packed[:, :H, :] = Y
    packed[:, H:, :hw] = Cb
    packed[:, H:, hw:] = Cr
    del Y, Cb, Cr

    return packed  # stays on GPU


def _yuv420p_gpu_to_cpu(packed: torch.Tensor) -> list[np.ndarray]:
    """Transfer YUV420P packed tensor from GPU → list of numpy arrays for encoder."""
    packed_np = packed.cpu().numpy()
    N = packed_np.shape[0]
    del packed
    return [packed_np[i] for i in range(N)]


def _yuv420p_to_rgb_gpu(
    yuv_packed: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert YUV420P packed tensor [N, H*3//2, W] uint8 → RGB uint8 [N, 3, H, W] on GPU.

    Input layout (per frame): Y plane (H rows, W cols) then [Cb|Cr] (H/2 rows, W cols)
    where Cb occupies left W/2 cols and Cr occupies right W/2 cols.

    All math runs on GPU. Returns tensor in CHW format ready for the pipeline.
    """
    global _BT709_INV
    if _BT709_INV.device != device:
        _BT709_INV = _BT709_INV.to(device)

    N, total_h, W = yuv_packed.shape
    H = (total_h * 2) // 3
    hh = H // 2
    hw = W // 2

    # Upload to GPU if not already there
    if yuv_packed.device != device:
        yuv_packed = yuv_packed.to(device, non_blocking=True)

    # Extract planes
    Y  = yuv_packed[:, :H, :].float()          # [N, H, W]
    Cb = yuv_packed[:, H:, :hw].float() - 128.0  # [N, hh, hw]
    Cr = yuv_packed[:, H:, hw:].float() - 128.0  # [N, hh, hw]

    # Upsample chroma to full resolution (nearest neighbor = fast)
    # [N, hh, hw] → [N, H, W]
    Cb = Cb.unsqueeze(1)  # [N, 1, hh, hw]
    Cr = Cr.unsqueeze(1)  # [N, 1, hh, hw]
    Cb = F.interpolate(Cb, size=(H, W), mode='nearest').squeeze(1)  # [N, H, W]
    Cr = F.interpolate(Cr, size=(H, W), mode='nearest').squeeze(1)  # [N, H, W]

    # Stack into [N, H, W, 3] for matmul with [3, 3] inverse matrix
    yuv = torch.stack([Y, Cb, Cr], dim=-1)  # [N, H, W, 3]
    del Y, Cb, Cr

    # YUV→RGB: rgb = yuv @ _BT709_INV^T
    rgb = (yuv @ _BT709_INV.T).clamp_(0, 255).to(torch.uint8)  # [N, H, W, 3]
    del yuv

    # HWC → CHW
    return rgb.permute(0, 3, 1, 2)  # [N, 3, H, W] uint8


# ── BT.709 forward matrix for delta conversion (RGB delta → YUV delta) ──────
# For additive delta (zero-centered), no offset needed:
#   dY  =  0.2126*dR + 0.7152*dG + 0.0722*dB
#   dCb = -0.1146*dR - 0.3854*dG + 0.5000*dB
#   dCr =  0.5000*dR - 0.4542*dG - 0.0458*dB
_BT709_FWD = torch.tensor([
    [ 0.2126,  0.7152,  0.0722],  # Y
    [-0.1146, -0.3854,  0.5000],  # Cb
    [ 0.5000, -0.4542, -0.0458],  # Cr
], dtype=torch.float32)  # [3, 3]


def _yuv420p_resize_down_gpu(
    yuv_packed: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Resize YUV420P packed tensor on GPU: [N, H*3//2, W] → [N, th*3//2, tw].
    Resizes Y, Cb, Cr planes independently (bilinear for Y, bilinear for chroma).
    """
    N, total_h, W = yuv_packed.shape
    H = (total_h * 2) // 3
    hh, hw = H // 2, W // 2
    th, tw = target_h - target_h % 2, target_w - target_w % 2
    thh, thw = th // 2, tw // 2

    Y  = yuv_packed[:, :H, :].unsqueeze(1).float()       # [N, 1, H, W]
    Cb = yuv_packed[:, H:, :hw].unsqueeze(1).float()      # [N, 1, hh, hw]
    Cr = yuv_packed[:, H:, hw:].unsqueeze(1).float()      # [N, 1, hh, hw]

    Y_s  = F.interpolate(Y,  size=(th, tw),   mode='bilinear', align_corners=False).to(torch.uint8)
    Cb_s = F.interpolate(Cb, size=(thh, thw), mode='bilinear', align_corners=False).to(torch.uint8)
    Cr_s = F.interpolate(Cr, size=(thh, thw), mode='bilinear', align_corners=False).to(torch.uint8)

    packed = torch.empty((N, th + thh, tw), dtype=torch.uint8, device=yuv_packed.device)
    packed[:, :th, :]     = Y_s.squeeze(1)
    packed[:, th:, :thw]  = Cb_s.squeeze(1)
    packed[:, th:, thw:]  = Cr_s.squeeze(1)
    return packed


def _get_encoder_config(bitrate_kbps: int = 0) -> dict:
    """Return encoder codec name and stream options based on NVENC availability.
    
    Args:
        bitrate_kbps: target video bitrate in kbps. If >0, use VBR to match source bitrate.
    """
    if _NVENC_AVAILABLE:
        if bitrate_kbps > 0:
            return {
                "codec": "h264_nvenc",
                "options": {
                    "rc": "vbr",
                    "b": f"{bitrate_kbps}k",
                    "maxrate": f"{int(bitrate_kbps * 1.2)}k",
                    "bufsize": f"{bitrate_kbps * 2}k",
                    "preset": NVENC_PRESET,
                },
            }
        else:
            return {
                "codec": "h264_nvenc",
                "options": {"rc": "constqp", "qp": CRF, "preset": NVENC_PRESET},
            }
    else:
        _x264_preset = ENCODE_PRESET or "ultrafast"
        if bitrate_kbps > 0:
            return {
                "codec": "h264",
                "options": {
                    "b": f"{bitrate_kbps}k",
                    "maxrate": f"{int(bitrate_kbps * 1.2)}k",
                    "bufsize": f"{bitrate_kbps * 2}k",
                    "preset": _x264_preset,
                },
            }
        else:
            return {
                "codec": "h264",
                "options": {"crf": CRF, "preset": _x264_preset},
            }


def _probe_bitrate(path: str) -> int:
    """Get video bitrate in kbps from a file. Returns 0 on failure."""
    try:
        with av.open(path) as c:
            vs = c.streams.video[0]
            # PyAV exposes bit_rate on the stream or codec context
            br = vs.bit_rate or vs.codec_context.bit_rate
            if br and br > 0:
                return int(br / 1000)
            # Fallback: estimate from file size and duration
            duration = float(vs.duration * vs.time_base) if vs.duration else 0
            if duration > 0:
                fsize_bits = os.path.getsize(path) * 8
                return int(fsize_bits / duration / 1000)
    except Exception:
        pass
    return 0

_INTERP = dict(mode="bilinear", align_corners=False, antialias=True)


def _probe_video_info(path: str) -> dict:
    """Probe video info via ffprobe. Returns {fps, width, height, n_frames, codec}."""
    import subprocess, json
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", path,
        ]
        out = subprocess.check_output(cmd, timeout=10)
        info = json.loads(out)["streams"][0]
        # Parse fps from r_frame_rate "50/1" or avg_frame_rate
        fps_str = info.get("r_frame_rate", info.get("avg_frame_rate", "24/1"))
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 24.0
        return {
            "fps": fps,
            "width": int(info.get("width", 0)),
            "height": int(info.get("height", 0)),
            "n_frames": int(info.get("nb_frames", 0)),
            "codec": info.get("codec_name", ""),
        }
    except Exception:
        return {"fps": 24.0, "width": 0, "height": 0, "n_frames": 0, "codec": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_video_model(ckpt_path: str, device: torch.device):
    """Load custom VideoSeal checkpoint with optional chunk_size override."""
    model = setup_model_from_checkpoint(ckpt_path)
    
    # Override chunk_size if specified in environment
    custom_chunk_size = os.getenv("MODEL_CHUNK_SIZE")
    if custom_chunk_size:
        original_chunk = model.chunk_size
        model.chunk_size = int(custom_chunk_size)
        print(f"Model chunk_size overridden: {original_chunk} -> {model.chunk_size}")
    
    model.eval().to(device)
    return model


def load_audio_model(device: torch.device):
    """Load AudioSeal generator (16-bit). Returns None if unavailable."""
    import logging
    _log = logging.getLogger(__name__)
    try:
        from audioseal import AudioSeal
        gen = AudioSeal.load_generator("audioseal_wm_16bits").eval().to(device)
        return gen
    except Exception as e:
        _log.warning(f"AudioSeal load failed: {type(e).__name__}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_audio_msg(text: str, device: torch.device) -> torch.Tensor:
    """Encode first 2 ASCII chars of text into a 16-bit tensor for AudioSeal."""
    txt2 = (text + "\x00\x00")[:2]
    bits = []
    for ch in txt2:
        b = ord(ch) & 0xFF
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return torch.tensor(bits, dtype=torch.float32, device=device).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Standard embed (original — full-res to GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_batch_standard(frames: torch.Tensor, model, msg_gpu, device) -> torch.Tensor:
    """Embed a batch of frames. Auto-splits on OOM (recursive halving).
    Sends full-resolution frames to GPU."""
    import logging, time as _t
    _log = logging.getLogger(__name__)

    video_gpu = None
    outputs = None
    try:
        _t0 = _t.perf_counter()
        video_gpu = frames.to(device)
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        _t_upload = _t.perf_counter() - _t0

        _t0 = _t.perf_counter()
        with torch.no_grad():
            with _cuda_lock:
                outputs = model.embed(video_gpu, msgs=msg_gpu, is_video=True)
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        _t_model = _t.perf_counter() - _t0

        _t0 = _t.perf_counter()
        result = outputs["imgs_w"].cpu()
        _t_download = _t.perf_counter() - _t0

        _log.info(
            f"  [TIMING-STD] {frames.shape[0]}f {frames.shape[2]}x{frames.shape[3]}  "
            f"upload={_t_upload:.3f}s  model={_t_model:.3f}s  download={_t_download:.3f}s  "
            f"total={_t_upload+_t_model+_t_download:.3f}s"
        )

        del video_gpu, outputs
        return result
    except torch.cuda.OutOfMemoryError:
        del video_gpu, outputs
        torch.cuda.empty_cache()

        global _oom_splits
        _oom_splits += 1

        n = frames.shape[0]
        if n <= 1:
            raise  # single frame still OOM → resolution too high
        half = n // 2
        _log.warning(f"  OOM at {n} frames, splitting → {half}+{n - half}")
        part1 = _embed_batch_standard(frames[:half], model, msg_gpu, device)
        part2 = _embed_batch_standard(frames[half:], model, msg_gpu, device)
        return torch.cat([part1, part2], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Fast embed (Option A — resize on CPU, only 256×256 to GPU)
# ─────────────────────────────────────────────────────────────────────────────

def _get_watermark_delta(frames_256: torch.Tensor, model, msg_gpu, device,
                         keep_on_gpu: bool = False) -> torch.Tensor:
    """Run the model's embedder on 256×256 frames and return the watermark delta.
    All GPU work happens at 256×256 — minimal VRAM and PCIe transfer.

    Args:
        frames_256: [N, 3, 256, 256] pre-resized on CPU
        model: Videoseal model (embedder, blender, attenuation on GPU)
        msg_gpu: message tensor on GPU
        device: GPU device
        keep_on_gpu: if True, return delta on GPU (avoids extra CPU→GPU transfer for GPU blend)

    Returns:
        delta_256: [N, 3, 256, 256] watermark delta on CPU (or GPU if keep_on_gpu)
    """
    import time as _t
    import logging
    _log = logging.getLogger(__name__)

    chunk_size = model.chunk_size   # internal model chunk (e.g. 32)
    step_size  = model.step_size    # key frame interval (e.g. 4)

    N = frames_256.shape[0]
    msgs = msg_gpu[:1].repeat(chunk_size, 1)  # match model's expected batch

    all_deltas = []
    _t_upload_total = 0.0
    _t_embedder_total = 0.0
    _t_vidmode_total = 0.0
    _t_atten_total = 0.0
    _t_download_total = 0.0

    for ii in range(0, len(frames_256[::step_size]), chunk_size):
        nimgs_in_ck = min(chunk_size, len(frames_256[::step_size]) - ii)
        start = ii * step_size
        end = start + nimgs_in_ck * step_size
        ck_256 = frames_256[start:end]  # [f, 3, 256, 256]

        # Adjust msgs for last chunk
        ck_msgs = msgs[:nimgs_in_ck] if nimgs_in_ck < chunk_size else msgs

        # Send only 256×256 to GPU (non-blocking for overlap)
        _t0 = _t.perf_counter()
        ck_gpu = ck_256.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        _t_upload_total += _t.perf_counter() - _t0

        # Sample key frames
        key_frames = ck_gpu[::step_size]  # [n, 3, 256, 256]

        # YUV conversion if model uses it
        if model.embedder.yuv:
            key_frames = model.rgb2yuv(key_frames)[:, 0:1]

        # Embedder → watermark delta at 256×256
        _t0 = _t.perf_counter()
        with torch.no_grad():
            with _cuda_lock:
                if USE_FP16 and device.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        preds_w = model.embedder(key_frames, ck_msgs.to(device))
                else:
                    preds_w = model.embedder(key_frames, ck_msgs.to(device))
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        _t_embedder_total += _t.perf_counter() - _t0

        # Video mode expansion (repeat/interpolate across chunk)
        _t0 = _t.perf_counter()
        preds_w = model._apply_video_mode(preds_w, len(ck_256), step_size, model.video_mode)
        _t_vidmode_total += _t.perf_counter() - _t0

        # Low-res attenuation (at 256×256 — cheap)
        _t0 = _t.perf_counter()
        if model.attenuation is not None:
            model.attenuation.to(device)
            hmaps = model.attenuation.heatmaps(ck_gpu)
            preds_w = hmaps * preds_w
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        _t_atten_total += _t.perf_counter() - _t0

        _t0 = _t.perf_counter()
        if keep_on_gpu:
            all_deltas.append(preds_w)
        else:
            all_deltas.append(preds_w.cpu())
        _t_download_total += _t.perf_counter() - _t0

        del ck_gpu
        if not keep_on_gpu:
            del preds_w

    _log.info(
        f"  [TIMING-DELTA] {N}f 256x256  "
        f"upload={_t_upload_total:.3f}s  embedder={_t_embedder_total:.3f}s  "
        f"vidmode={_t_vidmode_total:.3f}s  atten={_t_atten_total:.3f}s  "
        f"download={_t_download_total:.3f}s  "
        f"total={_t_upload_total+_t_embedder_total+_t_vidmode_total+_t_atten_total+_t_download_total:.3f}s"
    )

    return torch.cat(all_deltas, dim=0)[:N]


def _embed_batch_fast(frames: torch.Tensor, model, msg_gpu, device) -> torch.Tensor:
    """Fast embed: CPU resize → GPU at 256×256 → CPU/GPU blend (configurable).
    Transfers ~70× less data over PCIe for 2K video.
    
    GPU_BLEND=true: Move resize_up + blend to GPU (~26% faster, +500MB GPU)
    GPU_BLEND=false: Original CPU blend
    """
    import logging, time as _t
    _log = logging.getLogger(__name__)
    global _oom_splits

    N, C, H, W = frames.shape
    img_size = model.img_size  # 256

    if GPU_BLEND and device.type == "cuda":
        # ── GPU-EARLY PATH ─
        _t_alloc = 0.0
        _t_transfer = 0.0

        if frames.is_cuda:
            # Already on GPU (from _gpu_worker pipeline) — skip upload entirely
            frames_gpu = frames
            _t_upload = 0.0
        else:
            # Upload from CPU via preallocated GPU buffer
            _t0 = _t.perf_counter()
            _cache = _embed_batch_fast.__dict__
            _key = "gpu_upload_buf"
            gpu_buf = _cache.get(_key)
            if gpu_buf is None or gpu_buf.shape != frames.shape or gpu_buf.dtype != frames.dtype:
                gpu_buf = torch.empty(frames.shape, dtype=frames.dtype, device=device)
                _cache[_key] = gpu_buf
            _t_alloc = _t.perf_counter() - _t0

            _t0 = _t.perf_counter()
            gpu_buf.copy_(frames, non_blocking=True)
            torch.cuda.current_stream(device).synchronize()
            _t_transfer = _t.perf_counter() - _t0
            _t_upload = _t_alloc + _t_transfer
            frames_gpu = gpu_buf

        # Normalize on GPU (uint8 → float32, /255)
        _t0 = _t.perf_counter()
        _cache = _embed_batch_fast.__dict__
        _fkey = "gpu_float_buf"
        float_buf = _cache.get(_fkey)
        if float_buf is None or float_buf.shape[:2] != (N, C) or float_buf.shape[2:] != (H, W):
            float_buf = torch.empty((N, C, H, W), dtype=torch.float32, device=device)
            _cache[_fkey] = float_buf
        float_buf.copy_(frames_gpu)  # uint8 → float32 (implicit cast)
        float_buf.div_(255.0)
        frames_gpu = float_buf
        _t_normalize = _t.perf_counter() - _t0

        # Resize down on GPU (much faster than CPU)
        _t0 = _t.perf_counter()
        if H != img_size or W != img_size:
            frames_256 = F.interpolate(frames_gpu, size=(img_size, img_size), **_INTERP)
        else:
            frames_256 = frames_gpu
        _t_resize_down = _t.perf_counter() - _t0

        del frames  # free CPU memory early

        # 2. GPU: get watermark delta at 256×256 (frames_256 already on GPU)
        _t0 = _t.perf_counter()
        try:
            delta_256 = _get_watermark_delta(frames_256, model, msg_gpu, device,
                                              keep_on_gpu=True)
        except torch.cuda.OutOfMemoryError:
            _oom_splits += 1
            if N <= 1:
                raise
            half = N // 2
            _log.warning(f"  OOM at {N} frames (fast-gpu), splitting → {half}+{N - half}")
            # Fallback: need original CPU frames — reconstruct from GPU
            frames_cpu = frames_gpu.cpu()
            del frames_gpu
            part1 = _embed_batch_fast(frames_cpu[:half], model, msg_gpu, device)
            part2 = _embed_batch_fast(frames_cpu[half:], model, msg_gpu, device)
            return torch.cat([part1, part2], dim=0)
        _t_delta = _t.perf_counter() - _t0

        del frames_256

        # 3-5. Blend on GPU (frames_gpu already there — no second upload!)
        _t0 = _t.perf_counter()
        if H != img_size or W != img_size:
            delta_full_gpu = F.interpolate(delta_256, size=(H, W), **_INTERP)
        else:
            delta_full_gpu = delta_256
        del delta_256

        scaling_i = float(model.blender.scaling_i)
        scaling_w = float(model.blender.scaling_w)
        delta_full_gpu.mul_(scaling_w)
        frames_gpu.mul_(scaling_i).add_(delta_full_gpu)
        del delta_full_gpu

        if model.clamp:
            frames_gpu.clamp_(0, 1)

        torch.cuda.current_stream(device).synchronize()
        _t_blend = _t.perf_counter() - _t0

        _log.info(
            f"  [TIMING-FAST] {N}f {H}x{W}  "
            f"gpu_alloc={_t_alloc:.3f}s  gpu_transfer={_t_transfer:.3f}s  "
            f"gpu_normalize={_t_normalize:.3f}s  "
            f"gpu_resize_down={_t_resize_down:.3f}s  "
            f"delta={_t_delta:.3f}s  resize_up+blend={_t_blend:.3f}s  "
            f"total={_t_upload+_t_normalize+_t_resize_down+_t_delta+_t_blend:.3f}s"
        )
        return frames_gpu

    # ── CPU resize_down path (GPU_BLEND=false or no CUDA) ────────────
    # 1. CPU: resize down to 256×256
    _t0 = _t.perf_counter()
    if H != img_size or W != img_size:
        frames_256 = F.interpolate(frames, size=(img_size, img_size), **_INTERP)
    else:
        frames_256 = frames
    _t_resize_down = _t.perf_counter() - _t0

    # 2. GPU: get watermark delta at 256×256 (tiny transfer)
    _t0 = _t.perf_counter()
    try:
        delta_256 = _get_watermark_delta(frames_256, model, msg_gpu, device,
                                          keep_on_gpu=GPU_BLEND)
    except torch.cuda.OutOfMemoryError:
        # Fallback: split in half
        _oom_splits += 1
        if N <= 1:
            raise
        half = N // 2
        _log.warning(f"  OOM at {N} frames (fast), splitting → {half}+{N - half}")
        part1 = _embed_batch_fast(frames[:half], model, msg_gpu, device)
        part2 = _embed_batch_fast(frames[half:], model, msg_gpu, device)
        return torch.cat([part1, part2], dim=0)
    _t_delta = _t.perf_counter() - _t0

    del frames_256

    # 3-5. CPU PATH: resize_up + blend on CPU
    _t0 = _t.perf_counter()
    # 3. CPU: resize delta back to original resolution
    if H != img_size or W != img_size:
        delta_full = F.interpolate(delta_256, size=(H, W), **_INTERP)
    else:
        delta_full = delta_256
    del delta_256

    # 4. CPU: blend
    result = model.blender.scaling_i * frames + model.blender.scaling_w * delta_full
    del delta_full

    # 5. Clamp
    if model.clamp:
        result = torch.clamp(result, 0, 1)

    _t_blend = _t.perf_counter() - _t0

    _log.info(
        f"  [TIMING-FAST] {N}f {H}x{W}  "
        f"resize_down={_t_resize_down:.3f}s  delta={_t_delta:.3f}s  "
        f"cpu_resize_up+blend={_t_blend:.3f}s  "
        f"total={_t_resize_down+_t_delta+_t_blend:.3f}s"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Flush chunk (writes embedded frames to output container)
# ─────────────────────────────────────────────────────────────────────────────

def _flush_chunk(
    chunk: list,
    model,
    msg_gpu: torch.Tensor,
    device: torch.device,
    out_container,
    out_stream,
    fps: float,
    h_out,
    w_out,
    fast: bool = False,
    bitrate_kbps: int = 0,
) -> tuple:
    import time as _t, logging
    _log = logging.getLogger(__name__)

    _t0 = _t.perf_counter()
    # For fast+GPU_BLEND: pass uint8 directly — normalize happens on GPU inside _embed_batch_fast
    if fast and GPU_BLEND:
        video_chunk = torch.stack(chunk)
    else:
        video_chunk = torch.stack(chunk).float().div_(255.0)
    chunk.clear()  # free raw frame memory immediately
    _t_stack = _t.perf_counter() - _t0

    _t0 = _t.perf_counter()
    if fast:
        wm_chunk = _embed_batch_fast(video_chunk, model, msg_gpu, device)
    else:
        wm_chunk = _embed_batch_standard(video_chunk, model, msg_gpu, device)
    _t_embed = _t.perf_counter() - _t0

    del video_chunk

    _t0 = _t.perf_counter()
    if wm_chunk.is_cuda:
        rgb_uint8 = wm_chunk.clamp_(0, 1).mul_(255).to(torch.uint8).permute(0, 2, 3, 1)
    else:
        rgb_uint8 = (wm_chunk.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1)
        if device.type == "cuda":
            rgb_uint8 = rgb_uint8.to(device)
    del wm_chunk
    _t_clamp = _t.perf_counter() - _t0

    _t0 = _t.perf_counter()
    yuv_frames = _rgb_to_yuv420p_gpu(rgb_uint8)
    n_frames = rgb_uint8.shape[0]
    del rgb_uint8
    _t_yuv = _t.perf_counter() - _t0

    _t0 = _t.perf_counter()
    if out_stream is None:
        _yuv_h, _yuv_w = yuv_frames[0].shape
        H = (_yuv_h * 2) // 3
        W = _yuv_w
        H = H - H % 2
        W = W - W % 2
        enc_cfg = _get_encoder_config(bitrate_kbps)
        out_stream         = out_container.add_stream(enc_cfg["codec"], rate=int(fps))
        out_stream.width   = W
        out_stream.height  = H
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = enc_cfg["options"]
        # Enable multi-threaded encoding for libx264
        if enc_cfg["codec"] == "h264":
            out_stream.codec_context.thread_count = min(4, os.cpu_count() or 2)
            out_stream.codec_context.thread_type = av.codec.context.ThreadType.FRAME
        h_out, w_out = H, W

    _t_from_ndarray = 0.0
    _t_encode_hw = 0.0
    _t_mux_write = 0.0
    for yuv_np in yuv_frames:
        _t1 = _t.perf_counter()
        av_frame = av.VideoFrame.from_ndarray(yuv_np, format="yuv420p")
        _t_from_ndarray += _t.perf_counter() - _t1

        _t1 = _t.perf_counter()
        packets = out_stream.encode(av_frame)
        _t_encode_hw += _t.perf_counter() - _t1

        _t1 = _t.perf_counter()
        for packet in packets:
            out_container.mux(packet)
        _t_mux_write += _t.perf_counter() - _t1
    _t_encode = _t.perf_counter() - _t0

    _log.info(
        f"  [TIMING-FLUSH] {n_frames}f  "
        f"stack={_t_stack:.3f}s  embed={_t_embed:.3f}s  "
        f"clamp+uint8={_t_clamp:.3f}s  rgb2yuv_gpu={_t_yuv:.3f}s  "
        f"from_ndarray={_t_from_ndarray:.3f}s  encode={_t_encode_hw:.3f}s  mux={_t_mux_write:.3f}s  "
        f"encode_total={_t_encode:.3f}s  "
        f"total={_t_stack+_t_embed+_t_clamp+_t_yuv+_t_encode:.3f}s"
    )

    return out_stream, h_out, w_out


# ─────────────────────────────────────────────────────────────────────────────
# Sequential embed (original — decode→embed→encode one chunk at a time)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_sequential(
    av_path: str,
    tmp_vid: str,
    video_model,
    msg_gpu: torch.Tensor,
    device: torch.device,
    use_fast: bool,
    progress_callback,
    filename: str,
    mode_str: str,
    bitrate_kbps: int = 0,
) -> tuple:
    """Sequential: decode→embed→encode per chunk. Returns (fps, total_frames, res)."""
    import logging, time as _time
    _log = logging.getLogger(__name__)

    fps = 24.0
    total_frames = 0
    h_out = w_out = None
    out_stream = None

    with av.open(tmp_vid, mode="w") as out_container:
        with av.open(av_path) as in_container:
            in_stream = in_container.streams.video[0]
            rate = in_stream.average_rate or in_stream.guessed_rate
            if rate:
                fps = float(rate)
            n_total = in_stream.frames or 0
            res = f"{in_stream.width}x{in_stream.height}"
            _log.info(f"[EMBED-{mode_str}] {filename}: {res}, ~{n_total}f, fps={fps:.1f}, chunk={CHUNK_SIZE}, bitrate={bitrate_kbps}kbps")
            in_stream.thread_type = "AUTO"
            in_stream.codec_context.thread_count = DECODE_THREADS or min(8, os.cpu_count() or 4)

            if progress_callback and n_total > 0:
                progress_callback(0, n_total)

            chunk: list = []
            for frame in in_container.decode(in_stream):
                chunk.append(
                    torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                )
                if len(chunk) >= CHUNK_SIZE:
                    t0 = _time.time()
                    out_stream, h_out, w_out = _flush_chunk(
                        chunk, video_model, msg_gpu, device,
                        out_container, out_stream, fps, h_out, w_out,
                        fast=use_fast,
                        bitrate_kbps=bitrate_kbps,
                    )
                    total_frames += CHUNK_SIZE
                    dt = _time.time() - t0
                    gpu_s = ""
                    if device.type == "cuda":
                        alloc = torch.cuda.memory_allocated(device) / 1024**2
                        peak  = torch.cuda.max_memory_allocated(device) / 1024**2
                        util  = f"alloc={alloc:.0f}MB peak={peak:.0f}MB"
                        gpu_s = f"  [{util}]"
                    _log.info(f"  [{filename}] {total_frames}/{n_total}f  ({dt:.1f}s, {CHUNK_SIZE/dt:.1f}f/s){gpu_s}")
                    chunk = []

                    if progress_callback and n_total > 0:
                        progress_callback(total_frames, n_total)

            if chunk:
                n_remaining = len(chunk)
                out_stream, h_out, w_out = _flush_chunk(
                    chunk, video_model, msg_gpu, device,
                    out_container, out_stream, fps, h_out, w_out,
                    fast=use_fast,
                    bitrate_kbps=bitrate_kbps,
                )
                total_frames += n_remaining

                if progress_callback and n_total > 0:
                    progress_callback(total_frames, n_total)

        if out_stream:
            for packet in out_stream.encode():
                out_container.mux(packet)

    return fps, total_frames, res


# ─────────────────────────────────────────────────────────────────────────────
# Pipelined embed (3-stage: decode | GPU embed | H.264 encode)
# ─────────────────────────────────────────────────────────────────────────────

def _embed_pipelined(
    av_path: str,
    tmp_vid: str,
    video_model,
    msg_gpu: torch.Tensor,
    device: torch.device,
    use_fast: bool,
    progress_callback,
    filename: str,
    mode_str: str,
    bitrate_kbps: int = 0,
) -> tuple:
    """
    3-stage pipeline: decode | GPU embed | H.264 encode in parallel threads.
    GPU never waits for decode or encode — up to ~40% faster than sequential.
    Returns (fps, total_frames, res).
    """
    import logging
    import queue
    import threading
    import time as _time
    _log = logging.getLogger(__name__)

    SENTINEL = object()
    decode_q = queue.Queue(maxsize=2)   # decoded chunks waiting for GPU (reduced from 3)
    encode_q = queue.Queue(maxsize=2)   # embedded frames waiting for encode (reduced from 4)
    stop     = threading.Event()
    errors   = []
    state    = {"total_frames": 0}

    # ── Stage 2: GPU embed thread ─────────────────────────────────────────
    def _gpu_worker():
        """
        GPU embed thread — YUV-native pipeline:
          1. Upload YUV packed → GPU
          2. Resize_down YUV → 256×256 on GPU
          3. yuv2rgb at 256×256 (tiny — ~0.2ms)
          4. Embedder → delta RGB 256×256
          5. Convert delta RGB→YUV at 256×256 (~0.2ms)
          6. Resize_up delta YUV → full-res
          7. Blend delta onto original YUV full-res
          8. Download YUV packed → CPU

        Eliminates full-res yuv2rgb (135ms) + rgb2yuv (66ms) = 201ms/chunk.
        """
        try:
            compute_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
            _gpu_yuv_buf = [None, None]
            _gpu_buf_idx = 0
            _pinned_buf = None

            # Cache BT.709 forward matrix on device
            global _BT709_FWD
            if _BT709_FWD.device != device:
                _BT709_FWD = _BT709_FWD.to(device)

            while not stop.is_set():
                try:
                    item = decode_q.get(timeout=2)
                except queue.Empty:
                    continue
                if item is SENTINEL:
                    encode_q.put(SENTINEL)
                    return

                yuv_batch = item  # uint8 tensor [N, H*3//2, W] pinned CPU
                _t0_iter = _time.perf_counter()

                if compute_stream:
                    with torch.cuda.stream(compute_stream):
                        # ── 1. Upload YUV packed → GPU ──
                        _t0 = _time.perf_counter()
                        idx = _gpu_buf_idx
                        gpu_yuv = _gpu_yuv_buf[idx]
                        if gpu_yuv is None or gpu_yuv.shape != yuv_batch.shape:
                            gpu_yuv = torch.empty(yuv_batch.shape, dtype=torch.uint8, device=device)
                            _gpu_yuv_buf[idx] = gpu_yuv
                        gpu_yuv.copy_(yuv_batch, non_blocking=True)
                        _gpu_buf_idx = 1 - idx
                        del yuv_batch
                        compute_stream.synchronize()
                        _t_upload = _time.perf_counter() - _t0

                        # Parse YUV dimensions
                        N, total_h, W = gpu_yuv.shape
                        H = (total_h * 2) // 3
                        img_size = video_model.img_size  # 256

                        # ── 2. Resize_down YUV → 256×256 on GPU ──
                        _t0 = _time.perf_counter()
                        yuv_256 = _yuv420p_resize_down_gpu(gpu_yuv, img_size, img_size)
                        compute_stream.synchronize()
                        _t_resize_down = _time.perf_counter() - _t0

                        # ── 3. yuv2rgb at 256×256 (tiny) ──
                        _t0 = _time.perf_counter()
                        rgb_256 = _yuv420p_to_rgb_gpu(yuv_256, device)  # [N, 3, 256, 256] uint8
                        del yuv_256
                        # Normalize to float32 [0, 1]
                        rgb_256_f = rgb_256.float().div_(255.0)
                        del rgb_256
                        compute_stream.synchronize()
                        _t_yuv2rgb = _time.perf_counter() - _t0

                        # ── 4. Embedder → delta 256×256 ──
                        _t0 = _time.perf_counter()
                        delta_256 = _get_watermark_delta(
                            rgb_256_f, video_model, msg_gpu, device, keep_on_gpu=True
                        )
                        del rgb_256_f
                        _t_embed = _time.perf_counter() - _t0

                        # ── 5. Convert delta → YUV planes at 256×256 ──
                        _t0 = _time.perf_counter()
                        scaling_w = float(video_model.blender.scaling_w)
                        delta_256.mul_(scaling_w)
                        dC = delta_256.shape[1]  # 1 if model.embedder.yuv, 3 if RGB

                        if dC == 1:
                            # Model outputs Y-only delta — no color conversion needed
                            dY  = delta_256                                    # [N, 1, 256, 256]
                            dCb = torch.zeros_like(dY)
                            dCr = torch.zeros_like(dY)
                        else:
                            # Model outputs RGB delta — convert to YUV
                            # NCHW → NHWC for matmul
                            d_hwc = delta_256.permute(0, 2, 3, 1)             # [N, 256, 256, 3]
                            d_yuv = d_hwc @ _BT709_FWD.T                     # [N, 256, 256, 3]
                            del d_hwc
                            dY  = d_yuv[..., 0].unsqueeze(1)                  # [N, 1, 256, 256]
                            dCb = d_yuv[..., 1].unsqueeze(1)                  # [N, 1, 256, 256]
                            dCr = d_yuv[..., 2].unsqueeze(1)                  # [N, 1, 256, 256]
                            del d_yuv
                        del delta_256
                        compute_stream.synchronize()
                        _t_delta2yuv = _time.perf_counter() - _t0

                        # ── 6. Resize_up delta YUV planes → full-res ──
                        _t0 = _time.perf_counter()
                        hh, hw = H // 2, W // 2
                        _interp = dict(mode="bilinear", align_corners=False)
                        dY_full  = F.interpolate(dY,  size=(H, W), **_interp)    # [N, 1, H, W]
                        if dC > 1:
                            dCb_full = F.interpolate(dCb, size=(hh, hw), **_interp)
                            dCr_full = F.interpolate(dCr, size=(hh, hw), **_interp)
                        del dY, dCb, dCr
                        compute_stream.synchronize()
                        _t_resize_up = _time.perf_counter() - _t0

                        # ── 7. Blend delta onto original YUV full-res ──
                        _t0 = _time.perf_counter()
                        scaling_i = float(video_model.blender.scaling_i)

                        # Y plane: always blend
                        Y_orig = gpu_yuv[:, :H, :].float()
                        Y_out  = (scaling_i * Y_orig + dY_full.squeeze(1)).clamp_(0, 255).to(torch.uint8)
                        del Y_orig, dY_full

                        n_frames = N
                        out_packed = gpu_yuv.clone()  # start from original (preserves chroma)
                        out_packed[:, :H, :] = Y_out
                        del Y_out

                        # Cb/Cr planes: only blend if delta has chroma
                        if dC > 1:
                            Cb_orig = gpu_yuv[:, H:, :hw].float()
                            Cr_orig = gpu_yuv[:, H:, hw:].float()
                            Cb_out = (scaling_i * Cb_orig + dCb_full.squeeze(1)).clamp_(0, 255).to(torch.uint8)
                            Cr_out = (scaling_i * Cr_orig + dCr_full.squeeze(1)).clamp_(0, 255).to(torch.uint8)
                            out_packed[:, H:, :hw] = Cb_out
                            out_packed[:, H:, hw:] = Cr_out
                            del Cb_orig, Cr_orig, Cb_out, Cr_out, dCb_full, dCr_full

                        compute_stream.synchronize()
                        _t_blend = _time.perf_counter() - _t0

                        # ── 8. Download YUV packed → CPU ──
                        _t0 = _time.perf_counter()
                        if _pinned_buf is None or _pinned_buf.shape != out_packed.shape:
                            _pinned_buf = torch.empty(
                                out_packed.shape, dtype=torch.uint8, device='cpu',
                            ).pin_memory()
                        _pinned_buf.copy_(out_packed, non_blocking=True)
                        del out_packed
                        compute_stream.synchronize()
                        _t_download = _time.perf_counter() - _t0

                    # Convert to numpy list for encoder
                    packed_np = _pinned_buf[:n_frames].numpy()
                    np_list = [packed_np[i] for i in range(n_frames)]

                else:
                    # No CUDA — fallback to old path
                    _t_upload = _t_resize_down = _t_yuv2rgb = _t_embed = 0
                    _t_delta2yuv = _t_resize_up = _t_blend = _t_download = 0
                    frames_batch = _yuv420p_to_rgb_gpu(yuv_batch, device)
                    del yuv_batch
                    if use_fast:
                        wm = _embed_batch_fast(frames_batch, video_model, msg_gpu, device)
                    else:
                        video_chunk = frames_batch.float().div_(255.0)
                        del frames_batch
                        wm = _embed_batch_standard(video_chunk, video_model, msg_gpu, device)
                        del video_chunk
                    if wm.is_cuda:
                        rgb_uint8 = wm.clamp_(0, 1).mul_(255).to(torch.uint8).permute(0, 2, 3, 1)
                    else:
                        rgb_uint8 = (wm.clamp(0, 1).mul_(255)).to(torch.uint8).permute(0, 2, 3, 1)
                    n_frames = rgb_uint8.shape[0]
                    del wm
                    packed = _rgb_to_yuv420p_gpu_compute(rgb_uint8)
                    del rgb_uint8
                    packed_cpu = packed.cpu()
                    del packed
                    packed_np = packed_cpu[:n_frames].numpy()
                    np_list = [packed_np[i] for i in range(n_frames)]

                _t_iter = _time.perf_counter() - _t0_iter
                _log.info(
                    f"  [TIMING-PIPE-GPU] {n_frames}f  "
                    f"upload={_t_upload:.3f}s  resize_dn={_t_resize_down:.3f}s  "
                    f"yuv2rgb={_t_yuv2rgb:.3f}s  embed={_t_embed:.3f}s  "
                    f"d2yuv={_t_delta2yuv:.3f}s  resize_up={_t_resize_up:.3f}s  "
                    f"blend={_t_blend:.3f}s  download={_t_download:.3f}s  "
                    f"iter={_t_iter:.3f}s"
                )

                encode_q.put(np_list)

        except Exception as e:
            errors.append(e)
            stop.set()
            try:
                encode_q.put(SENTINEL)
            except Exception:
                pass

    # ── Stage 3: H.264 encode thread ──────────────────────────────────────
    def _enc_worker(tmp_vid_path, fps):
        """Encode via FFmpeg subprocess pipe — avoids PyAV from_ndarray overhead."""
        import subprocess
        ffmpeg_proc = None
        stderr_thread = None
        stderr_lines = []
        H = W = 0

        def _drain_stderr(proc, lines_buf):
            """Read FFmpeg stderr in background to prevent pipe deadlock."""
            try:
                for line in proc.stderr:
                    decoded = line.decode(errors="replace").rstrip()
                    if decoded:
                        lines_buf.append(decoded)
            except Exception:
                pass

        try:
            while not stop.is_set():
                try:
                    item = encode_q.get(timeout=2)
                except queue.Empty:
                    continue
                if item is SENTINEL:
                    break

                vid_np = item  # list of YUV420P numpy arrays [H*3//2, W] uint8
                n = len(vid_np)

                # Lazy-start FFmpeg on first chunk (need H, W)
                if ffmpeg_proc is None:
                    _yuv_h, _yuv_w = vid_np[0].shape
                    H = (_yuv_h * 2) // 3
                    W = _yuv_w
                    H = H - H % 2
                    W = W - W % 2

                    enc_cfg = _get_encoder_config(bitrate_kbps)
                    codec = enc_cfg["codec"]
                    opts = enc_cfg["options"]

                    # Map codec name: PyAV "h264" → FFmpeg "libx264"
                    if codec == "h264":
                        codec = "libx264"

                    cmd = [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                        "-f", "rawvideo",
                        "-pix_fmt", "yuv420p",
                        "-s", f"{W}x{H}",
                        "-r", str(int(fps)),
                        "-i", "pipe:0",
                        "-c:v", codec,
                    ]
                    # Add encoder options
                    for k, v in opts.items():
                        if k == "b":
                            cmd += ["-b:v", str(v)]
                        elif k == "maxrate":
                            cmd += ["-maxrate", str(v)]
                        elif k == "bufsize":
                            cmd += ["-bufsize", str(v)]
                        else:
                            cmd += [f"-{k}", str(v)]
                    cmd += ["-pix_fmt", "yuv420p", tmp_vid_path]

                    _log.info(f"  [ENC-PIPE] Starting FFmpeg: {' '.join(cmd)}")
                    ffmpeg_proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    # Start stderr drain thread to prevent deadlock
                    stderr_thread = threading.Thread(
                        target=_drain_stderr,
                        args=(ffmpeg_proc, stderr_lines),
                        daemon=True, name="ffmpeg-stderr",
                    )
                    stderr_thread.start()

                # Write raw YUV bytes to FFmpeg stdin
                _t0_all = _time.perf_counter()
                _t_tobytes = 0.0
                _t_write = 0.0
                for yuv_np_frame in vid_np:
                    _t1 = _time.perf_counter()
                    raw = yuv_np_frame.tobytes()
                    _t_tobytes += _time.perf_counter() - _t1

                    _t1 = _time.perf_counter()
                    ffmpeg_proc.stdin.write(raw)
                    _t_write += _time.perf_counter() - _t1
                _t_enc = _time.perf_counter() - _t0_all

                _log.info(
                    f"  [TIMING-PIPE-ENC] {n}f  "
                    f"tobytes={_t_tobytes:.3f}s  pipe_write={_t_write:.3f}s  "
                    f"total={_t_enc:.3f}s  ({n/_t_enc:.1f}f/s)"
                )

                state["total_frames"] += n
                if progress_callback:
                    progress_callback(state["total_frames"], 0)

            # Close stdin to signal EOF → FFmpeg finishes encoding
            if ffmpeg_proc and ffmpeg_proc.stdin:
                _log.info("  [ENC-PIPE] Closing stdin, waiting for FFmpeg to finish...")
                _t0_flush = _time.perf_counter()
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait(timeout=300)
                _t_flush = _time.perf_counter() - _t0_flush
                _log.info(f"  [ENC-PIPE] FFmpeg finished in {_t_flush:.1f}s")
                if stderr_thread:
                    stderr_thread.join(timeout=5)
                if ffmpeg_proc.returncode != 0:
                    stderr_tail = "\n".join(stderr_lines[-20:])
                    _log.error(f"  [ENC-PIPE] FFmpeg failed (rc={ffmpeg_proc.returncode}):\n{stderr_tail}")
                    raise RuntimeError(f"FFmpeg encode failed: rc={ffmpeg_proc.returncode}")
                elif stderr_lines:
                    _log.info(f"  [ENC-PIPE] FFmpeg warnings: {stderr_lines[-3:]}")
        except Exception as e:
            errors.append(e)
            stop.set()
        finally:
            if ffmpeg_proc:
                try:
                    if ffmpeg_proc.stdin and not ffmpeg_proc.stdin.closed:
                        ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait(timeout=30)
                except Exception:
                    ffmpeg_proc.kill()
                    ffmpeg_proc.wait(timeout=5)

    # ── Stage 1: decode (main thread) + orchestrate ───────────────────────
    fps = 24.0
    res = "?"
    n_total = 0

    with av.open(av_path) as in_container:
        in_stream = in_container.streams.video[0]
        rate = in_stream.average_rate or in_stream.guessed_rate
        if rate:
            fps = float(rate)
        n_total = in_stream.frames or 0
        res = f"{in_stream.width}x{in_stream.height}"
        in_stream.thread_type = "AUTO"
        in_stream.codec_context.thread_count = DECODE_THREADS or min(8, os.cpu_count() or 4)

        _log.info(
            f"[EMBED-{mode_str}] {filename}: {res}, "
            f"~{n_total}f, fps={fps:.1f}, chunk={CHUNK_SIZE}"
        )

        if progress_callback and n_total > 0:
            progress_callback(0, n_total)

        # Start worker threads
        gpu_t = threading.Thread(target=_gpu_worker, daemon=True, name="pipe-gpu")
        enc_t = threading.Thread(
            target=_enc_worker, args=(tmp_vid, fps),
            daemon=True, name="pipe-enc",
        )
        gpu_t.start()
        enc_t.start()

        t_start = _time.time()
        chunk = []
        chunks_decoded = 0
        _t_avdec_total = 0.0
        _t_ndarray_total = 0.0
        _t_tensor_total = 0.0
        _t_decode_total = 0.0
        _t_stack_total = 0.0

        # Preallocated pinned decode buffers (lazy init after first frame gives shape)
        _dec_buf_a = None  # double buffer A
        _dec_buf_b = None  # double buffer B
        _dec_use_a = True  # toggle
        _frame_shape = None  # (H*3//2, W) for YUV420P

        _frame_iter = in_container.decode(in_stream)
        frame_idx_in_chunk = 0
        while not stop.is_set():
            _t0 = _time.perf_counter()
            try:
                frame = next(_frame_iter)
            except StopIteration:
                break
            _t1 = _time.perf_counter()
            _t_avdec_total += _t1 - _t0

            # Decode to native YUV420P (no swscale conversion — near zero-cost)
            nd = frame.to_ndarray(format="yuv420p")  # shape [H*3//2, W] uint8
            _t2 = _time.perf_counter()
            _t_ndarray_total += _t2 - _t1

            # Lazy-init preallocated pinned buffers on first frame
            if _frame_shape is None:
                _frame_shape = nd.shape  # (H*3//2, W)
                _dec_buf_a = torch.empty(
                    (CHUNK_SIZE, *_frame_shape), dtype=torch.uint8,
                ).pin_memory()
                _dec_buf_b = torch.empty(
                    (CHUNK_SIZE, *_frame_shape), dtype=torch.uint8,
                ).pin_memory()
                _log.info(f"  [PREALLOC] decode buffers: 2x [{CHUNK_SIZE}, {_frame_shape[0]}, {_frame_shape[1]}] pinned "
                          f"({2 * _dec_buf_a.nbytes / 1024**2:.0f}MB total)")

            # Copy frame into preallocated buffer slot (avoids per-frame tensor alloc)
            cur_dec_buf = _dec_buf_a if _dec_use_a else _dec_buf_b
            cur_dec_buf[frame_idx_in_chunk].numpy()[:] = nd
            frame_idx_in_chunk += 1
            _t3 = _time.perf_counter()
            _t_tensor_total += _t3 - _t2
            _t_decode_total += _t3 - _t0

            if frame_idx_in_chunk >= CHUNK_SIZE:
                _t0 = _time.perf_counter()
                batch = cur_dec_buf[:frame_idx_in_chunk]  # view, no copy
                _t_stack_total += _time.perf_counter() - _t0

                _dec_use_a = not _dec_use_a  # swap to other buffer
                frame_idx_in_chunk = 0

                while not stop.is_set():
                    try:
                        decode_q.put(batch, timeout=2)
                        break
                    except queue.Full:
                        continue
                chunks_decoded += 1

                decoded_f = chunks_decoded * CHUNK_SIZE
                dt = _time.time() - t_start
                gpu_s = ""
                if device.type == "cuda":
                    alloc = torch.cuda.memory_allocated(device) / 1024**2
                    peak  = torch.cuda.max_memory_allocated(device) / 1024**2
                    gpu_s = f"  [alloc={alloc:.0f}MB peak={peak:.0f}MB]"
                _log.info(
                    f"  [{filename}] decode={decoded_f} embed={state['total_frames']} "
                    f"/ {n_total}f  ({dt:.1f}s){gpu_s}"
                )
                _log.info(
                    f"  [TIMING-PIPE-DEC] chunk#{chunks_decoded}  "
                    f"av_decode={_t_avdec_total:.3f}s  to_ndarray={_t_ndarray_total:.3f}s  "
                    f"to_tensor={_t_tensor_total:.3f}s  stack+pin={_t_stack_total:.3f}s  "
                    f"total={_t_decode_total+_t_stack_total:.3f}s"
                )
                _t_decode_total = 0.0
                _t_avdec_total = 0.0
                _t_ndarray_total = 0.0
                _t_tensor_total = 0.0
                _t_stack_total = 0.0

                if progress_callback and n_total > 0:
                    progress_callback(min(decoded_f, n_total), n_total)

        # Send remaining frames
        if frame_idx_in_chunk > 0 and not stop.is_set():
            cur_dec_buf = _dec_buf_a if _dec_use_a else _dec_buf_b
            batch = cur_dec_buf[:frame_idx_in_chunk].clone().pin_memory()
            while not stop.is_set():
                try:
                    decode_q.put(batch, timeout=2)
                    break
                except queue.Full:
                    continue

        # Signal end of decode
        if not stop.is_set():
            while not stop.is_set():
                try:
                    decode_q.put(SENTINEL, timeout=2)
                    break
                except queue.Full:
                    continue
        else:
            try:
                decode_q.put(SENTINEL, timeout=1)
            except Exception:
                pass

    # Wait for workers
    gpu_t.join(timeout=600)
    enc_t.join(timeout=600)

    if errors:
        raise errors[0]

    # Final progress
    if progress_callback and n_total > 0:
        progress_callback(state["total_frames"], n_total)

    return fps, state["total_frames"], res


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline
# ─────────────────────────────────────────────────────────────────────────────

def embed_to_file(
    input_path: str,
    output_path: str,
    video_model,
    watermark_text: str,
    device: torch.device,
    audio_model=None,
    progress_callback=None,
    fast_embed: bool = None,
) -> dict:
    """
    Full pipeline: embed VideoSeal + (optionally) AudioSeal watermark.

    Args:
        input_path:       source video file
        output_path:      destination (same filename/ext expected by caller)
        video_model:      loaded VideoSeal custom model
        watermark_text:   text to embed (BCH-256 for video, first 2 chars for audio)
        device:           torch.device
        audio_model:      AudioSeal generator, or None to skip audio watermark
        progress_callback: optional callable(processed_frames, total_frames)
        fast_embed:       True=CPU resize pipeline, False=original, None=use global FAST_EMBED

    Returns:
        dict with keys: fps, total_frames, has_audio, resolution, oom_splits, mode
    """
    global _oom_splits
    _oom_splits = 0  # reset per video
    use_fast = FAST_EMBED if fast_embed is None else fast_embed

    # ── Encode watermark text → 256-bit BCH tensor ────────────────────────────
    msg_tensor, _codeword, _bits = text_to_msg_tensor_bch(watermark_text, msg_bits=MSG_BITS)

    ext     = Path(input_path).suffix.lstrip(".").lower()
    out_ext = Path(output_path).suffix.lstrip(".").lower()
    fmt     = _EXT_TO_FORMAT.get(out_ext, "mp4")

    # Pre-convert formats PyAV cannot read directly
    tmp_converted: Optional[str] = None
    av_path = input_path
    if ext in _FFMPEG_CONVERT_EXT:
        tmp_converted = _to_mp4_via_ffmpeg(input_path)
        av_path = tmp_converted

    import logging, time as _time
    _log = logging.getLogger(__name__)

    # ── Extract original audio ────────────────────────────────────────────────
    _t0_audio = _time.time()
    audio_orig  = extract_audio_track(input_path)   # None → no audio track
    has_audio   = audio_orig is not None
    audio_final = audio_orig                         # may be replaced by wm'd version

    # ── Optionally watermark audio ────────────────────────────────────────────
    if has_audio and audio_model is not None:
        try:
            import torchaudio
            import torchaudio.functional as TAF

            wav_orig, sr_orig = torchaudio.load(audio_orig)
            n_channels_orig = wav_orig.shape[0]

            # Downmix + resample to 16kHz mono for AudioSeal
            wav = wav_orig.clone()
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr_orig != 16000:
                wav = TAF.resample(wav, sr_orig, 16000)
            wav = wav.float().to(device)

            msg16 = _encode_audio_msg(watermark_text, device)  # [1, 16]
            with torch.no_grad():
                wm_wav = audio_model(
                    wav.unsqueeze(0), sample_rate=16000, message=msg16
                ).squeeze(0).cpu()

            # Resample back to original sample rate
            if sr_orig != 16000:
                wm_wav = TAF.resample(wm_wav, 16000, sr_orig)
            # Restore original channel count (duplicate mono → stereo if needed)
            if n_channels_orig > 1 and wm_wav.shape[0] == 1:
                wm_wav = wm_wav.expand(n_channels_orig, -1)

            tmp_afd, tmp_audio_wm = tempfile.mkstemp(suffix=".wav")
            os.close(tmp_afd)
            torchaudio.save(tmp_audio_wm, wm_wav, sr_orig)
            audio_final = tmp_audio_wm
        except Exception:
            # Fall back to original audio — video watermark still proceeds
            audio_final = audio_orig

    # ── Embed video frames ────────────────────────────────────────────────────
    tmp_vfd, tmp_vid = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_vfd)

    _t_audio_total = _time.time() - _t0_audio
    _log.info(f"[TIMING] {Path(input_path).name}: audio_extract+wm={_t_audio_total:.3f}s (has_audio={has_audio})")

    mode_str = "FAST" if use_fast else "STANDARD"
    if use_fast and GPU_BLEND:
        mode_str += "+GPU-BLEND"
    if _NVENC_AVAILABLE:
        mode_str += "+NVENC"
    if USE_FP16:
        mode_str += "+FP16"
    if PIPELINE:
        mode_str += "+PIPE"
    msg_gpu   = msg_tensor.to(device)
    _filename = Path(input_path).name

    # ── Probe source bitrate to match in output ──────────────────────────────
    bitrate_kbps = _probe_bitrate(av_path)
    if bitrate_kbps > 0:
        _log.info(f"[BITRATE] {_filename}: source={bitrate_kbps} kbps, will use VBR to match")
        mode_str += f"+VBR({bitrate_kbps}k)"
    else:
        _log.info(f"[BITRATE] {_filename}: could not detect, using CRF={CRF}")

    # ── Auto-adjust chunk size based on available VRAM ────────────────────
    effective_chunk = _auto_chunk_size(CHUNK_SIZE, device)
    if effective_chunk < CHUNK_SIZE:
        _log.warning(f"[AUTO-CHUNK] {_filename}: reduced chunk {CHUNK_SIZE}→{effective_chunk} (low VRAM)")
        # Temporarily override global CHUNK_SIZE for this video
        import worker as _worker_mod
        _orig_chunk = _worker_mod.CHUNK_SIZE
        _worker_mod.CHUNK_SIZE = effective_chunk
    else:
        _orig_chunk = None

    try:
        _t0_video = _time.time()
        if PIPELINE:
            fps, total_frames, res = _embed_pipelined(
                av_path, tmp_vid, video_model, msg_gpu, device,
                use_fast, progress_callback, _filename, mode_str,
                bitrate_kbps=bitrate_kbps,
            )
        else:
            fps, total_frames, res = _embed_sequential(
                av_path, tmp_vid, video_model, msg_gpu, device,
                use_fast, progress_callback, _filename, mode_str,
                bitrate_kbps=bitrate_kbps,
            )
        _t_video_total = _time.time() - _t0_video
        _log.info(f"[TIMING] {_filename}: video_embed={_t_video_total:.3f}s ({total_frames}f, {total_frames/_t_video_total:.1f}f/s)")
    finally:
        del msg_gpu
        if tmp_converted and os.path.exists(tmp_converted):
            os.unlink(tmp_converted)
        # Restore original chunk size if it was auto-adjusted
        if _orig_chunk is not None:
            import worker as _worker_mod
            _worker_mod.CHUNK_SIZE = _orig_chunk

    # ── Mux audio + finalize container ───────────────────────────────────────
    import subprocess
    _t0_mux = _time.time()
    vid_moved = False
    try:
        if audio_final and os.path.exists(audio_final):
            _mux_video_audio(tmp_vid, audio_final, output_path, fmt)
        elif fmt != "mp4":
            from web_demo.core.video_io import _remux_video
            _remux_video(tmp_vid, output_path, fmt)
        else:
            if os.path.exists(output_path):
                os.unlink(output_path)
            shutil.move(tmp_vid, output_path)
            vid_moved = True
    finally:
        if not vid_moved and os.path.exists(tmp_vid):
            os.unlink(tmp_vid)
        if audio_orig and os.path.exists(audio_orig):
            os.unlink(audio_orig)
        if audio_final and audio_final != audio_orig and os.path.exists(audio_final):
            os.unlink(audio_final)

    _t_mux_total = _time.time() - _t0_mux
    _log.info(f"[TIMING] {_filename}: mux={_t_mux_total:.3f}s")

    return {
        "fps": fps,
        "total_frames": total_frames,
        "has_audio": has_audio,
        "resolution": res,
        "oom_splits": _oom_splits,
        "mode": mode_str,
    }
