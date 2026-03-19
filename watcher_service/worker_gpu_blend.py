"""
GPU-accelerated blend optimization for FAST mode.
Moves blending to GPU while keeping CPU resize for memory efficiency.
"""
import os
import torch
from torch.nn import functional as F

_INTERP = dict(mode="bilinear", align_corners=False, antialias=True)


def _embed_batch_fast_gpu_blend(frames: torch.Tensor, model, msg_gpu, device) -> torch.Tensor:
    """
    Fast embed with GPU BLEND optimization:
    - CPU resize down (save memory)
    - GPU inference at 256×256
    - GPU resize up (fast on GPU)
    - GPU blend (FAST! ~20× faster than CPU)
    - Return to CPU
    
    Expected speedup: ~26% (4.0s blend → 0.2s)
    GPU memory increase: ~500 MB
    """
    import logging
    _log = logging.getLogger(__name__)

    N, C, H, W = frames.shape
    img_size = model.img_size  # 256

    # 1. CPU: resize down to 256×256 (save memory)
    if H != img_size or W != img_size:
        frames_256 = F.interpolate(frames, size=(img_size, img_size), **_INTERP)
    else:
        frames_256 = frames

    # 2. GPU: get watermark delta at 256×256 (tiny transfer)
    from worker import _get_watermark_delta, _oom_splits
    
    try:
        delta_256 = _get_watermark_delta(frames_256, model, msg_gpu, device)
    except torch.cuda.OutOfMemoryError:
        # Fallback: split in half
        _oom_splits += 1
        if N <= 1:
            raise
        half = N // 2
        _log.warning(f"  OOM at {N} frames (fast GPU blend), splitting → {half}+{N - half}")
        part1 = _embed_batch_fast_gpu_blend(frames[:half], model, msg_gpu, device)
        part2 = _embed_batch_fast_gpu_blend(frames[half:], model, msg_gpu, device)
        return torch.cat([part1, part2], dim=0)

    del frames_256

    # 3-5. GPU: resize up + blend (OPTIMIZED!)
    # Transfer original frames to GPU once
    frames_gpu = frames.to(device, non_blocking=True)
    
    # 3. GPU: resize delta to full resolution
    if H != img_size or W != img_size:
        delta_full_gpu = F.interpolate(delta_256, size=(H, W), **_INTERP)
    else:
        delta_full_gpu = delta_256
    del delta_256

    # 4. GPU: blend (FAST - ~0.2s vs 4.0s on CPU!)
    result_gpu = model.blender.scaling_i * frames_gpu + model.blender.scaling_w * delta_full_gpu
    del frames_gpu, delta_full_gpu

    # 5. GPU: clamp
    if model.clamp:
        result_gpu = torch.clamp(result_gpu, 0, 1)

    # 6. Transfer result back to CPU
    result = result_gpu.cpu()
    del result_gpu

    return result


def _embed_batch_fast_full_gpu(frames: torch.Tensor, model, msg_gpu, device) -> torch.Tensor:
    """
    AGGRESSIVE GPU optimization: ALL operations on GPU
    - GPU resize down (fast)
    - GPU inference
    - GPU resize up (fast)
    - GPU blend (fast)
    - Return to CPU
    
    Expected speedup: ~50% total
    GPU memory increase: ~2 GB (full-res frames on GPU)
    Trade-off: May not support 4K on 16GB GPU
    """
    import logging
    _log = logging.getLogger(__name__)

    N, C, H, W = frames.shape
    img_size = model.img_size  # 256

    # Transfer original frames to GPU ONCE at start
    frames_gpu = frames.to(device, non_blocking=True)

    # 1. GPU: resize down to 256×256 (FAST on GPU!)
    if H != img_size or W != img_size:
        frames_256_gpu = F.interpolate(frames_gpu, size=(img_size, img_size), **_INTERP)
    else:
        frames_256_gpu = frames_gpu

    # 2. GPU: get watermark delta at 256×256
    # Note: _get_watermark_delta expects CPU input, so we need to modify it
    # For now, do inference inline
    from worker import _get_watermark_delta
    
    try:
        # Temporarily move to CPU for _get_watermark_delta
        frames_256_cpu = frames_256_gpu.cpu()
        delta_256 = _get_watermark_delta(frames_256_cpu, model, msg_gpu, device)
        del frames_256_cpu
        delta_256_gpu = delta_256.to(device)
        del delta_256
    except torch.cuda.OutOfMemoryError:
        # Fallback: split in half
        global _oom_splits
        _oom_splits += 1
        if N <= 1:
            raise
        half = N // 2
        _log.warning(f"  OOM at {N} frames (full GPU), splitting → {half}+{N - half}")
        part1 = _embed_batch_fast_full_gpu(frames[:half], model, msg_gpu, device)
        part2 = _embed_batch_fast_full_gpu(frames[half:], model, msg_gpu, device)
        return torch.cat([part1, part2], dim=0)

    del frames_256_gpu

    # 3. GPU: resize delta to full resolution (FAST!)
    if H != img_size or W != img_size:
        delta_full_gpu = F.interpolate(delta_256_gpu, size=(H, W), **_INTERP)
    else:
        delta_full_gpu = delta_256_gpu
    del delta_256_gpu

    # 4. GPU: blend (FAST!)
    result_gpu = model.blender.scaling_i * frames_gpu + model.blender.scaling_w * delta_full_gpu
    del frames_gpu, delta_full_gpu

    # 5. GPU: clamp
    if model.clamp:
        result_gpu = torch.clamp(result_gpu, 0, 1)

    # 6. Transfer result back to CPU
    result = result_gpu.cpu()
    del result_gpu

    return result


# Export functions
__all__ = ['_embed_batch_fast_gpu_blend', '_embed_batch_fast_full_gpu']
