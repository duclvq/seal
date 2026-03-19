"""
So sánh chất lượng video giữa các preset encode.
Đo: PSNR, SSIM, file size, encode speed.

Cách hoạt động CRF:
  - CRF (Constant Rate Factor) giữ chất lượng cảm nhận KHÔNG ĐỔI giữa các preset
  - Preset chỉ quyết định encoder dành bao nhiêu effort để nén
  - ultrafast: nén ít → file TO hơn, nhưng chất lượng BẰNG fast ở cùng CRF
  - fast: nén tốt hơn → file NHỎ hơn, chất lượng BẰNG ultrafast ở cùng CRF
  - Nói cách khác: cùng CRF 18, ultrafast và fast cho chất lượng gần như giống nhau,
    chỉ khác file size

Usage:
    python bench_quality.py --input ../data/val/gi-diploma-66013582898105388892353.mp4
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def decode_frames(path: str, max_frames: int = 300) -> tuple:
    """Decode video → numpy array [N,H,W,3] uint8 + fps."""
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "FRAME"
        fps = float(s.average_rate or s.guessed_rate or 25)
        w, h = s.width, s.height
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    return np.stack(frames), fps, w, h


def encode_video(frames_np: np.ndarray, fps: float, codec: str,
                 options: dict, out_path: str):
    """Encode numpy frames → video file."""
    N, H, W, C = frames_np.shape
    H2 = H - H % 2
    W2 = W - W % 2
    with av.open(out_path, mode="w") as out:
        stream = out.add_stream(codec, rate=int(fps))
        stream.width = W2
        stream.height = H2
        stream.pix_fmt = "yuv420p"
        stream.options = options
        for frame_np in frames_np[:, :H2, :W2, :]:
            f = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for pkt in stream.encode(f):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)


def decode_back(path: str, max_frames: int = 9999) -> np.ndarray:
    """Decode encoded video back to numpy for comparison."""
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "FRAME"
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= max_frames:
                break
    return np.stack(frames)


def calc_psnr(original: np.ndarray, encoded: np.ndarray) -> float:
    """Calculate PSNR between original and encoded frames."""
    n = min(len(original), len(encoded))
    orig = original[:n].astype(np.float64)
    enc = encoded[:n].astype(np.float64)
    mse = np.mean((orig - enc) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def calc_ssim_simple(original: np.ndarray, encoded: np.ndarray) -> float:
    """Simplified SSIM (per-frame mean luminance SSIM, averaged)."""
    n = min(len(original), len(encoded))
    ssim_vals = []
    for i in range(n):
        # Convert to grayscale
        orig_gray = np.mean(original[i].astype(np.float64), axis=2)
        enc_gray = np.mean(encoded[i].astype(np.float64), axis=2)

        mu_x = np.mean(orig_gray)
        mu_y = np.mean(enc_gray)
        sigma_x = np.std(orig_gray)
        sigma_y = np.std(enc_gray)
        sigma_xy = np.mean((orig_gray - mu_x) * (enc_gray - mu_y))

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
        ssim_vals.append(ssim)

    return np.mean(ssim_vals)


def main():
    parser = argparse.ArgumentParser(description="Quality comparison between presets")
    parser.add_argument("--input", required=True)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()

    log.info(f"Input: {args.input}")
    log.info(f"Frames: {args.frames}")

    # Decode original
    log.info("\nDecoding original...")
    orig_frames, fps, w, h = decode_frames(args.input, args.frames)
    n = len(orig_frames)
    log.info(f"  {n} frames, {w}x{h}, fps={fps:.1f}")

    # Test configs
    configs = [
        ("x264 ultrafast CRF18", "h264", {"preset": "ultrafast", "crf": "18"}),
        ("x264 veryfast  CRF18", "h264", {"preset": "veryfast", "crf": "18"}),
        ("x264 fast      CRF18", "h264", {"preset": "fast", "crf": "18"}),
        ("x264 medium    CRF18", "h264", {"preset": "medium", "crf": "18"}),
        ("x264 ultrafast CRF23", "h264", {"preset": "ultrafast", "crf": "23"}),
        ("x264 fast      CRF23", "h264", {"preset": "fast", "crf": "23"}),
    ]

    # Check NVENC
    try:
        av.codec.Codec("h264_nvenc", "w")
        configs.extend([
            ("NVENC p1  QP18", "h264_nvenc", {"preset": "p1", "rc": "constqp", "qp": "18"}),
            ("NVENC p4  QP18", "h264_nvenc", {"preset": "p4", "rc": "constqp", "qp": "18"}),
            ("NVENC p7  QP18", "h264_nvenc", {"preset": "p7", "rc": "constqp", "qp": "18"}),
        ])
    except Exception:
        pass

    log.info(f"\n{'='*80}")
    log.info(f"{'Preset':<28s} {'PSNR':>7s} {'SSIM':>7s} {'Size MB':>8s} "
             f"{'Ratio':>6s} {'FPS':>6s} {'Time':>6s}")
    log.info(f"{'='*80}")

    results = []
    for label, codec, opts in configs:
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        # Encode
        t0 = time.perf_counter()
        encode_video(orig_frames, fps, codec, opts, tmp)
        dt = time.perf_counter() - t0

        # File size
        sz_mb = os.path.getsize(tmp) / 1024**2
        orig_sz = orig_frames.nbytes / 1024**2  # raw uncompressed size

        # Decode back
        enc_frames = decode_back(tmp, n)

        # Quality metrics
        psnr = calc_psnr(orig_frames, enc_frames)
        ssim = calc_ssim_simple(orig_frames, enc_frames)

        enc_fps = n / dt
        ratio = sz_mb / orig_sz * 100  # % of raw size

        results.append({
            "label": label, "psnr": psnr, "ssim": ssim,
            "size_mb": sz_mb, "ratio": ratio, "fps": enc_fps, "time": dt,
        })

        log.info(f"{label:<28s} {psnr:7.2f} {ssim:7.4f} {sz_mb:8.1f} "
                 f"{ratio:5.1f}% {enc_fps:6.0f} {dt:6.2f}s")

        os.unlink(tmp)
        del enc_frames

    # Analysis
    log.info(f"\n{'='*80}")
    log.info("ANALYSIS")
    log.info(f"{'='*80}")

    # Compare ultrafast vs fast at CRF18
    uf = next((r for r in results if "ultrafast CRF18" in r["label"]), None)
    fa = next((r for r in results if r["label"].strip().startswith("x264 fast") and "CRF18" in r["label"]), None)

    if uf and fa:
        psnr_diff = uf["psnr"] - fa["psnr"]
        ssim_diff = uf["ssim"] - fa["ssim"]
        size_diff = uf["size_mb"] - fa["size_mb"]
        speed_ratio = uf["fps"] / fa["fps"]

        log.info(f"\n  ultrafast vs fast (CRF 18):")
        log.info(f"    PSNR:  {uf['psnr']:.2f} vs {fa['psnr']:.2f}  (diff: {psnr_diff:+.2f} dB)")
        log.info(f"    SSIM:  {uf['ssim']:.4f} vs {fa['ssim']:.4f}  (diff: {ssim_diff:+.4f})")
        log.info(f"    Size:  {uf['size_mb']:.1f} vs {fa['size_mb']:.1f} MB  ({size_diff:+.1f} MB, "
                 f"+{(uf['size_mb']/fa['size_mb']-1)*100:.0f}%)")
        log.info(f"    Speed: {uf['fps']:.0f} vs {fa['fps']:.0f} fps  ({speed_ratio:.1f}x faster)")

        if abs(psnr_diff) < 0.5 and abs(ssim_diff) < 0.005:
            log.info(f"\n  → Chất lượng KHÔNG KHÁC BIỆT đáng kể (PSNR diff < 0.5dB, SSIM diff < 0.005)")
            log.info(f"    ultrafast nhanh hơn {speed_ratio:.1f}x, file to hơn {(uf['size_mb']/fa['size_mb']-1)*100:.0f}%")
            log.info(f"    Với watermark service, file size không quan trọng bằng tốc độ → ultrafast OK")
        else:
            log.info(f"\n  → Có khác biệt chất lượng. Cân nhắc dùng veryfast thay ultrafast.")

    log.info(f"\n  Giải thích:")
    log.info(f"    - CRF mode: encoder tự điều chỉnh bitrate để giữ chất lượng cảm nhận không đổi")
    log.info(f"    - Preset chỉ quyết định encoder dành bao nhiêu CPU effort để tìm cách nén tối ưu")
    log.info(f"    - ultrafast: ít effort → file to hơn, nhưng chất lượng pixel gần như BẰNG")
    log.info(f"    - Khác biệt PSNR < 0.5dB = mắt người KHÔNG phân biệt được")
    log.info(f"    - Khác biệt SSIM < 0.005 = imperceptible")


if __name__ == "__main__":
    main()
