"""
Test: FFmpeg subprocess decoder for MXF files.
Verifies that watermark survives the full pipeline when decoding MXF
directly via ffmpeg (no pre-convert to MP4).

Usage:
    python watcher_service/tests/test_ffmpeg_decode_mxf.py
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_TESTS_DIR    = Path(__file__).resolve().parent
_SERVICE_DIR  = _TESTS_DIR.parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_SERVICE_DIR), str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.environ["FFMPEG_DECODE"] = "true"   # force ffmpeg decode path
os.chdir(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

import av
import torch
from worker import load_video_model, load_audio_model, embed_to_file, FFmpegDecoder
from web_demo.core.ecc import msg_tensor_to_text_bch

CKPT        = "output/run2_video/checkpoint350.pth"
INPUT_VIDEO = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
FIXTURES    = _TESTS_DIR / "fixtures"
WM_TEXT     = "test"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTRACT_CHUNK = 8
EXTRACT_FRAMES = 300

_cleanup = []


def _ffmpeg(args, timeout=120):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    log.info(f"  ffmpeg: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    return r.returncode == 0


def _ensure_mxf_fixture():
    """Create MXF fixture if not present (H.264-inside-MXF, 15s)."""
    mxf_path = str(FIXTURES / "test_clip.mxf")
    if os.path.exists(mxf_path):
        return mxf_path
    FIXTURES.mkdir(parents=True, exist_ok=True)
    ok = _ffmpeg(["-i", INPUT_VIDEO, "-t", "15", "-c:v", "libx264", "-crf", "18",
                  "-c:a", "pcm_s16le", mxf_path])
    if not ok:
        # Try MPEG-2 inside MXF as fallback
        ok = _ffmpeg(["-i", INPUT_VIDEO, "-t", "15", "-c:v", "mpeg2video", "-q:v", "2",
                      "-c:a", "pcm_s16le", "-f", "mxf", mxf_path])
    assert ok and os.path.exists(mxf_path), "Failed to create MXF fixture"
    return mxf_path


def _extract_video_watermark(video_path, model, max_frames=300):
    """Extract video watermark using streaming aggregation."""
    running_sum = None
    n_frames = 0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        chunk_buf = []
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            chunk_buf.append(torch.from_numpy(rgb).permute(2, 0, 1))
            if len(chunk_buf) >= EXTRACT_CHUNK:
                video_chunk = torch.stack(chunk_buf).to(DEVICE).float().div_(255.0)
                chunk_buf.clear()
                with torch.no_grad():
                    outputs = model.detect(video_chunk, is_video=True)
                preds = outputs["preds"]
                if preds.dim() == 4:
                    preds = preds.mean(dim=(-2, -1))
                bp = preds[:, 1:].cpu()
                cs = (bp * bp.abs()).sum(dim=0)
                running_sum = cs if running_sum is None else running_sum + cs
                n_frames += bp.shape[0]
                del video_chunk, outputs, preds, bp, cs
                torch.cuda.empty_cache()
            if n_frames >= max_frames:
                break
        if chunk_buf and n_frames < max_frames:
            video_chunk = torch.stack(chunk_buf).to(DEVICE).float().div_(255.0)
            chunk_buf.clear()
            with torch.no_grad():
                outputs = model.detect(video_chunk, is_video=True)
            preds = outputs["preds"]
            if preds.dim() == 4:
                preds = preds.mean(dim=(-2, -1))
            bp = preds[:, 1:].cpu()
            cs = (bp * bp.abs()).sum(dim=0)
            running_sum = cs if running_sum is None else running_sum + cs
            n_frames += bp.shape[0]
            del video_chunk, outputs, preds, bp, cs
    msg = (running_sum / n_frames > 0).unsqueeze(0)
    decode = msg_tensor_to_text_bch(msg)
    decode["n_frames_used"] = n_frames
    return decode


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_ffmpeg_decoder_reads_mxf():
    """FFmpegDecoder can iterate frames from MXF without error."""
    log.info("=" * 60)
    log.info("TEST 1: FFmpegDecoder reads MXF frames")
    log.info("=" * 60)
    mxf = _ensure_mxf_fixture()
    dec = FFmpegDecoder(mxf)
    log.info(f"  MXF: {dec.width}x{dec.height}, fps={dec.fps}, ~{dec.n_frames}f, codec={dec.codec_name}")
    count = 0
    for nd in dec:
        assert nd.shape == (dec.height * 3 // 2, dec.width), f"Bad shape: {nd.shape}"
        count += 1
        if count >= 30:
            break
    dec.close()
    log.info(f"  Read {count} frames OK")
    assert count >= 10, f"Expected >=10 frames, got {count}"
    return True


def test_mxf_embed_ffmpeg_decode():
    """Embed watermark into MXF using FFmpeg decode (no pre-convert). Verify WM preserved."""
    log.info("=" * 60)
    log.info("TEST 2: MXF embed via FFmpeg decode → watermark preserved")
    log.info("=" * 60)
    mxf_in = _ensure_mxf_fixture()
    model = load_video_model(CKPT, DEVICE)
    audio_model = load_audio_model(DEVICE)

    fd, out = tempfile.mkstemp(suffix=".mxf")
    os.close(fd)
    _cleanup.append(out)

    log.info(f"  Embedding: {mxf_in} -> {out}")
    t0 = time.time()
    result = embed_to_file(
        input_path=mxf_in, output_path=out,
        video_model=model, watermark_text=WM_TEXT,
        device=DEVICE, audio_model=audio_model,
    )
    t_embed = time.time() - t0
    log.info(f"  Embed done: {t_embed:.1f}s, {result['total_frames']}f, mode={result['mode']}")

    # Verify FFDEC was used (no pre-convert)
    assert "FFDEC" in result["mode"], f"Expected FFDEC in mode, got: {result['mode']}"

    # Verify output is valid
    assert os.path.exists(out) and os.path.getsize(out) > 0, "Output file empty"

    # Extract watermark
    log.info(f"  Extracting watermark from: {out}")
    decode = _extract_video_watermark(out, model, max_frames=EXTRACT_FRAMES)
    detected = decode.get("correctable", False)
    extracted = decode.get("decoded_text", "")
    match = extracted == WM_TEXT

    log.info(f"  Detected={detected}, text='{extracted}', match={match}, "
             f"frames_used={decode.get('n_frames_used', 0)}")

    assert detected, "Watermark not detected"
    assert match, f"Text mismatch: expected '{WM_TEXT}', got '{extracted}'"
    log.info("  PASSED")
    return True


def test_mxf_embed_output_mp4():
    """Embed MXF input → MP4 output. Verify WM preserved."""
    log.info("=" * 60)
    log.info("TEST 3: MXF input → MP4 output via FFmpeg decode")
    log.info("=" * 60)
    mxf_in = _ensure_mxf_fixture()
    model = load_video_model(CKPT, DEVICE)

    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    _cleanup.append(out)

    result = embed_to_file(
        input_path=mxf_in, output_path=out,
        video_model=model, watermark_text=WM_TEXT,
        device=DEVICE,
    )
    log.info(f"  Embed: {result['total_frames']}f, mode={result['mode']}")

    decode = _extract_video_watermark(out, model, max_frames=EXTRACT_FRAMES)
    detected = decode.get("correctable", False)
    extracted = decode.get("decoded_text", "")
    match = extracted == WM_TEXT

    log.info(f"  Detected={detected}, text='{extracted}', match={match}")
    assert detected and match, f"WM failed: detected={detected}, text='{extracted}'"
    log.info("  PASSED")
    return True


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    results = {}
    tests = [
        ("1_ffmpeg_decoder_reads_mxf", test_ffmpeg_decoder_reads_mxf),
        ("2_mxf_embed_ffmpeg_decode", test_mxf_embed_ffmpeg_decode),
        ("3_mxf_embed_output_mp4", test_mxf_embed_output_mp4),
    ]
    for name, fn in tests:
        try:
            passed = fn()
            results[name] = ("PASS" if passed else "FAIL", "")
        except Exception as e:
            log.error(f"  {name} ERROR: {e}", exc_info=True)
            results[name] = ("FAIL", str(e))

    # Cleanup
    for p in _cleanup:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

    # Summary
    print(f"\n{'='*60}")
    print("FFmpeg Decode MXF Test Results")
    print(f"{'='*60}")
    for name, (status, detail) in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}  {detail}")
    n_pass = sum(1 for s, _ in results.values() if s == "PASS")
    print(f"\n  {n_pass}/{len(results)} passed")
    print(f"{'='*60}")
    return n_pass == len(results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
