"""
Comprehensive test suite for the VideoSeal watcher pipeline.
Tests: watermark, quality, stability, flicker, frame count, file size,
       params, format support (MXF/WebM), extension preservation,
       and attack robustness (bitrate halving, crop 10%, horizontal flip).

Usage:
    python watcher_service/tests/test_pipeline.py
"""
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_TESTS_DIR    = Path(__file__).resolve().parent
_SERVICE_DIR  = _TESTS_DIR.parent
_PROJECT_ROOT = _SERVICE_DIR.parent
for p in [str(_SERVICE_DIR), str(_PROJECT_ROOT), str(_PROJECT_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.chdir(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

import av
import numpy as np
import torch

from worker import load_video_model, load_audio_model, embed_to_file
from web_demo.core.ecc import msg_tensor_to_text_bch

# ── Config ───────────────────────────────────────────────────────────────────
CKPT        = "output/run2_video/checkpoint350.pth"
INPUT_VIDEO = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
FIXTURES    = _TESTS_DIR / "fixtures"
WM_TEXT     = "test"
DEVICE      = torch.device("cuda")

MIN_PSNR_DB        = 35.0
MAX_FILESIZE_DIFF  = 0.05
MAX_FRAME_DROP     = 2
FLICKER_THRESHOLD  = 3.0
EXTRACT_FRAMES     = 300
EXTRACT_CHUNK      = 8

# ── Globals ──────────────────────────────────────────────────────────────────
_model = None
_audio_model = None
_output_path = None
_results = {}
_cleanup = []


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_models():
    global _model, _audio_model
    if _model is None:
        log.info("Loading video model...")
        _model = load_video_model(CKPT, DEVICE)
        log.info("Loading audio model...")
        _audio_model = load_audio_model(DEVICE)
        if _audio_model is None:
            log.warning("AudioSeal not available — audio watermark tests will be limited")
    return _model, _audio_model


def _embed_test_video() -> str:
    """Embed watermark into test video, return output path. Cached across tests."""
    global _output_path
    if _output_path and os.path.exists(_output_path):
        return _output_path
    model, audio_model = _load_models()
    fd, out = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    log.info(f"Embedding: {INPUT_VIDEO} -> {out}")
    t0 = time.time()
    result = embed_to_file(
        input_path=INPUT_VIDEO, output_path=out,
        video_model=model, watermark_text=WM_TEXT,
        device=DEVICE, audio_model=audio_model,
    )
    log.info(f"Embed done: {time.time()-t0:.1f}s, {result['total_frames']}f, mode={result['mode']}")
    _output_path = out
    _cleanup.append(out)
    return out


def _extract_video_watermark(video_path: str, model, max_frames: int = 300) -> dict:
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


def _probe_media_info(path: str) -> dict:
    """Probe video and audio parameters using PyAV."""
    info = {"video": {}, "audio": {}}
    with av.open(path) as c:
        if c.streams.video:
            vs = c.streams.video[0]
            rate = vs.average_rate or vs.guessed_rate
            info["video"] = {
                "width": vs.width, "height": vs.height,
                "fps": float(rate) if rate else 0,
                "n_frames": vs.frames or 0,
                "codec": vs.codec_context.name,
                "pix_fmt": vs.codec_context.pix_fmt,
                "duration": float(vs.duration * vs.time_base) if vs.duration else 0,
            }
        if c.streams.audio:
            aus = c.streams.audio[0]
            info["audio"] = {
                "sample_rate": aus.sample_rate, "channels": aus.channels,
                "codec": aus.codec_context.name,
                "duration": float(aus.duration * aus.time_base) if aus.duration else 0,
            }
    info["file_size"] = os.path.getsize(path)
    return info


def _ffmpeg(args: list, timeout: int = 120) -> bool:
    """Run ffmpeg command, return True on success."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning"] + args
    log.info(f"  ffmpeg: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        log.warning(f"  ffmpeg stderr: {r.stderr.decode(errors='replace')[-500:]}")
    return r.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Video watermark preserved
# ═══════════════════════════════════════════════════════════════════════════════
def test_video_watermark():
    log.info("=" * 60)
    log.info("TEST 1: Video watermark preservation")
    log.info("=" * 60)
    out = _embed_test_video()
    model, _ = _load_models()
    decode = _extract_video_watermark(out, model, max_frames=EXTRACT_FRAMES)
    detected = decode.get("correctable", False)
    extracted = decode.get("decoded_text", "")
    match = extracted == WM_TEXT
    log.info(f"  Detected={detected}, expected='{WM_TEXT}', extracted='{extracted}', match={match}")
    passed = detected and match
    _results["1_video_watermark"] = (passed, f"detected={detected}, text='{extracted}', match={match}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Audio watermark preserved
# ═══════════════════════════════════════════════════════════════════════════════
def test_audio_watermark():
    log.info("=" * 60)
    log.info("TEST 2: Audio watermark preservation")
    log.info("=" * 60)
    out = _embed_test_video()
    info = _probe_media_info(out)
    if not info["audio"]:
        _results["2_audio_watermark"] = (True, "skipped (no audio)")
        return True
    try:
        from audioseal import AudioSeal
        import torchaudio
        detector = AudioSeal.load_detector("audioseal_detector_16bits").eval().to(DEVICE)
        fd, tmp_audio = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        _cleanup.append(tmp_audio)
        subprocess.run([
            "ffmpeg", "-y", "-i", out, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", tmp_audio,
        ], capture_output=True, timeout=30)
        wav, sr = torchaudio.load(tmp_audio)
        wav = wav.float().to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            result = detector.detect_watermark(wav, sample_rate=sr, message_threshold=0.5)
        prob = float(result[0])
        extracted = ""
        if len(result) > 1 and result[1] is not None:
            bits = result[1].squeeze().cpu().numpy().astype(int).tolist()
            for ci in range(2):
                v = 0
                for bi in range(8):
                    v = (v << 1) | bits[ci * 8 + bi]
                extracted += chr(v & 0x7F)
        expected = (WM_TEXT + "\x00\x00")[:2]
        match = extracted == expected
        log.info(f"  prob={prob:.3f}, expected='{expected}', extracted='{extracted}', match={match}")
        passed = prob > 0.5 and match
        _results["2_audio_watermark"] = (passed, f"prob={prob:.3f}, text='{extracted}', match={match}")
        return passed
    except Exception as e:
        log.warning(f"  Audio watermark test error: {e}")
        _results["2_audio_watermark"] = (False, f"error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Video quality (PSNR ≥ 35dB)
# ═══════════════════════════════════════════════════════════════════════════════
def test_video_quality():
    log.info("=" * 60)
    log.info("TEST 3: Video quality (PSNR)")
    log.info("=" * 60)
    out = _embed_test_video()
    N = 200
    def _read(path, n):
        frames = []
        with av.open(path) as c:
            for i, f in enumerate(c.decode(c.streams.video[0])):
                if i >= n: break
                frames.append(f.to_ndarray(format="rgb24"))
        return frames
    inp_f = _read(INPUT_VIDEO, N)
    out_f = _read(out, N)
    n = min(len(inp_f), len(out_f))
    psnrs = []
    for i in range(n):
        mse = np.mean((inp_f[i].astype(np.float32) - out_f[i].astype(np.float32)) ** 2)
        psnrs.append(10 * np.log10(255**2 / mse) if mse > 0 else 99.0)
    mean_p = np.mean(psnrs)
    log.info(f"  {n} frames: mean={mean_p:.2f}dB, min={np.min(psnrs):.2f}dB, max={np.max(psnrs):.2f}dB")
    passed = mean_p >= MIN_PSNR_DB
    _results["3_video_quality"] = (passed, f"mean={mean_p:.2f}dB, min={np.min(psnrs):.2f}dB")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: File copy stability
# ═══════════════════════════════════════════════════════════════════════════════
def test_file_stability():
    log.info("=" * 60)
    log.info("TEST 4: File copy stability")
    log.info("=" * 60)
    from service import _is_file_stable
    import threading
    fd, tmp = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    try:
        # 4a: stable
        with open(tmp, "wb") as f: f.write(b"\x00" * 1024)
        a = _is_file_stable(Path(tmp), wait_seconds=1)
        log.info(f"  4a stable: {a} (expect True)")
        # 4b: empty
        with open(tmp, "wb") as f: pass
        b = not _is_file_stable(Path(tmp), wait_seconds=1)
        log.info(f"  4b empty:  {not b} → reject={b} (expect True)")
        # 4c: growing
        with open(tmp, "wb") as f: f.write(b"\x00" * 1024)
        def _grow():
            time.sleep(0.5)
            with open(tmp, "ab") as f: f.write(b"\x00" * 2048)
        t = threading.Thread(target=_grow, daemon=True); t.start()
        c = not _is_file_stable(Path(tmp), wait_seconds=2); t.join()
        log.info(f"  4c growing: reject={c} (expect True)")
        passed = a and b and c
        _results["4_file_stability"] = (passed, f"stable={a}, empty={b}, growing={c}")
        return passed
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: No flicker
# ═══════════════════════════════════════════════════════════════════════════════
def test_no_flicker():
    log.info("=" * 60)
    log.info("TEST 5: No flicker / frame jumping")
    log.info("=" * 60)
    out = _embed_test_video()
    frames = []
    with av.open(out) as c:
        for i, f in enumerate(c.decode(c.streams.video[0])):
            if i >= 200: break
            frames.append(f.to_ndarray(format="rgb24"))
    maes, jumps = [], []
    prev = frames[0].astype(np.float32)
    for i in range(1, len(frames)):
        cur = frames[i].astype(np.float32)
        mae = np.mean(np.abs(cur - prev))
        maes.append(mae)
        if len(maes) > 2:
            avg = np.mean(maes[max(0, len(maes)-11):len(maes)-1])
            if avg > 0 and mae > avg * FLICKER_THRESHOLD:
                jumps.append(i)
                log.warning(f"    JUMP frame {i}: MAE={mae:.3f}, avg={avg:.3f}")
        prev = cur
    log.info(f"  mean_mae={np.mean(maes):.3f}, max={np.max(maes):.3f}, jumps={len(jumps)}")
    passed = len(jumps) == 0
    _results["5_no_flicker"] = (passed, f"jumps={len(jumps)}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Frame count
# ═══════════════════════════════════════════════════════════════════════════════
def test_frame_count():
    log.info("=" * 60)
    log.info("TEST 6: Frame count preservation")
    log.info("=" * 60)
    out = _embed_test_video()
    inp_info = _probe_media_info(INPUT_VIDEO)
    out_info = _probe_media_info(out)
    inp_n = inp_info["video"].get("n_frames", 0)
    out_n = out_info["video"].get("n_frames", 0)
    if inp_n == 0:
        with av.open(INPUT_VIDEO) as c: inp_n = sum(1 for _ in c.decode(c.streams.video[0]))
    if out_n == 0:
        with av.open(out) as c: out_n = sum(1 for _ in c.decode(c.streams.video[0]))
    diff = abs(inp_n - out_n)
    log.info(f"  input={inp_n}, output={out_n}, diff={diff}, threshold≤{MAX_FRAME_DROP}")
    passed = diff <= MAX_FRAME_DROP
    _results["6_frame_count"] = (passed, f"input={inp_n}, output={out_n}, diff={diff}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: File size
# ═══════════════════════════════════════════════════════════════════════════════
def test_file_size():
    log.info("=" * 60)
    log.info("TEST 7: File size (HEVC→H.264 expected larger)")
    log.info("=" * 60)
    out = _embed_test_video()
    inp_sz = os.path.getsize(INPUT_VIDEO)
    out_sz = os.path.getsize(out)
    diff = abs(out_sz - inp_sz) / inp_sz
    log.info(f"  input={inp_sz/1024/1024:.1f}MB, output={out_sz/1024/1024:.1f}MB, diff={diff*100:.1f}%")
    # Note: HEVC→H.264 transcoding increases size ~50%, so this is informational
    passed = diff <= MAX_FILESIZE_DIFF
    _results["7_file_size"] = (passed, f"{inp_sz/1024/1024:.1f}MB→{out_sz/1024/1024:.1f}MB ({diff*100:.1f}%)")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 8: Params unchanged
# ═══════════════════════════════════════════════════════════════════════════════
def test_params_unchanged():
    log.info("=" * 60)
    log.info("TEST 8: Audio/video parameters preservation")
    log.info("=" * 60)
    out = _embed_test_video()
    inp = _probe_media_info(INPUT_VIDEO)
    outp = _probe_media_info(out)
    issues = []
    iv, ov = inp["video"], outp["video"]
    if iv["width"] != ov["width"] or iv["height"] != ov["height"]:
        issues.append(f"res: {iv['width']}x{iv['height']}→{ov['width']}x{ov['height']}")
    if abs(iv["fps"] - ov["fps"]) > 0.1:
        issues.append(f"fps: {iv['fps']:.2f}→{ov['fps']:.2f}")
    if abs(iv["duration"] - ov["duration"]) > 1.0:
        issues.append(f"dur: {iv['duration']:.2f}→{ov['duration']:.2f}s")
    log.info(f"  Res: {iv['width']}x{iv['height']}→{ov['width']}x{ov['height']}, "
             f"FPS: {iv['fps']:.2f}→{ov['fps']:.2f}, Codec: {iv['codec']}→{ov['codec']}")
    ia, oa = inp.get("audio", {}), outp.get("audio", {})
    if ia and oa:
        if ia["sample_rate"] != oa["sample_rate"]: issues.append(f"sr: {ia['sample_rate']}→{oa['sample_rate']}")
        if ia["channels"] != oa["channels"]: issues.append(f"ch: {ia['channels']}→{oa['channels']}")
        if abs(ia["duration"] - oa["duration"]) > 1.0: issues.append("audio dur mismatch")
        log.info(f"  Audio: SR={ia['sample_rate']}→{oa['sample_rate']}, CH={ia['channels']}→{oa['channels']}")
    elif ia and not oa:
        issues.append("audio track missing")
    if issues: log.warning(f"  Issues: {issues}")
    passed = len(issues) == 0
    _results["8_params_unchanged"] = (passed, f"{issues}" if issues else "all match")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 9: MXF format support
# ═══════════════════════════════════════════════════════════════════════════════
def test_mxf_format():
    log.info("=" * 60)
    log.info("TEST 9: MXF format embed + extract")
    log.info("=" * 60)
    mxf_in = str(FIXTURES / "test_clip.mxf")
    if not os.path.exists(mxf_in):
        log.info("  Creating MXF fixture from source video (15s H.264-inside-MXF)...")
        FIXTURES.mkdir(parents=True, exist_ok=True)
        _ffmpeg(["-i", INPUT_VIDEO, "-t", "15", "-c:v", "libx264", "-crf", "18",
                 "-c:a", "pcm_s16le", mxf_in])
    if not os.path.exists(mxf_in):
        _results["9_mxf_format"] = (False, "fixture creation failed")
        return False
    model, audio_model = _load_models()
    fd, out = tempfile.mkstemp(suffix=".mxf"); os.close(fd)
    _cleanup.append(out)
    try:
        result = embed_to_file(mxf_in, out, model, WM_TEXT, DEVICE, audio_model)
        log.info(f"  Embed: {result['total_frames']}f, mode={result['mode']}")
        # Verify output is valid video
        info = _probe_media_info(out)
        has_video = info["video"].get("width", 0) > 0
        # Extract watermark — need 300 frames for reliable MXF detection
        decode = _extract_video_watermark(out, model, max_frames=300)
        detected = decode.get("correctable", False)
        extracted = decode.get("decoded_text", "")
        match = extracted == WM_TEXT
        log.info(f"  Valid video={has_video}, detected={detected}, text='{extracted}', match={match}")
        passed = has_video and detected and match
        _results["9_mxf_format"] = (passed, f"video={has_video}, wm={detected}, match={match}")
        return passed
    except Exception as e:
        log.error(f"  MXF test error: {e}", exc_info=True)
        _results["9_mxf_format"] = (False, f"error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 10: WebM format support
# ═══════════════════════════════════════════════════════════════════════════════
def test_webm_format():
    log.info("=" * 60)
    log.info("TEST 10: WebM format embed + extract")
    log.info("=" * 60)
    webm_in = str(FIXTURES / "test_clip.webm")
    if not os.path.exists(webm_in):
        log.info("  Creating WebM fixture from source video (15s clip)...")
        FIXTURES.mkdir(parents=True, exist_ok=True)
        _ffmpeg(["-i", INPUT_VIDEO, "-t", "15", "-c:v", "libvpx-vp9", "-b:v", "5M",
                 "-c:a", "libopus", webm_in])
    if not os.path.exists(webm_in):
        _results["10_webm_format"] = (False, "fixture creation failed")
        return False
    model, audio_model = _load_models()
    fd, out = tempfile.mkstemp(suffix=".webm"); os.close(fd)
    _cleanup.append(out)
    try:
        result = embed_to_file(webm_in, out, model, WM_TEXT, DEVICE, audio_model)
        log.info(f"  Embed: {result['total_frames']}f, mode={result['mode']}")
        info = _probe_media_info(out)
        has_video = info["video"].get("width", 0) > 0
        decode = _extract_video_watermark(out, model, max_frames=300)
        detected = decode.get("correctable", False)
        extracted = decode.get("decoded_text", "")
        match = extracted == WM_TEXT
        log.info(f"  Valid video={has_video}, detected={detected}, text='{extracted}', match={match}")
        passed = has_video and detected and match
        _results["10_webm_format"] = (passed, f"video={has_video}, wm={detected}, match={match}")
        return passed
    except Exception as e:
        log.error(f"  WebM test error: {e}", exc_info=True)
        _results["10_webm_format"] = (False, f"error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 11: Output extension preserved
# ═══════════════════════════════════════════════════════════════════════════════
def test_extension_preserved():
    log.info("=" * 60)
    log.info("TEST 11: Output file extension preserved")
    log.info("=" * 60)
    model, audio_model = _load_models()
    issues = []

    for ext in [".mp4", ".mkv", ".mov", ".mxf", ".webm"]:
        # Find or create a fixture for this extension
        fixture = str(FIXTURES / f"test_clip{ext}")
        if not os.path.exists(fixture):
            FIXTURES.mkdir(parents=True, exist_ok=True)
            if ext == ".mxf":
                _ffmpeg(["-i", INPUT_VIDEO, "-t", "15", "-c:v", "libx264", "-crf", "18",
                         "-c:a", "pcm_s16le", fixture])
            elif ext == ".webm":
                _ffmpeg(["-i", INPUT_VIDEO, "-t", "15", "-c:v", "libvpx-vp9", "-b:v", "5M",
                         "-c:a", "libopus", fixture])
            else:
                _ffmpeg(["-i", INPUT_VIDEO, "-t", "3", "-c:v", "libx264", "-crf", "23",
                         "-c:a", "aac", fixture])

        if not os.path.exists(fixture):
            issues.append(f"{ext}: fixture failed")
            continue

        fd, out = tempfile.mkstemp(suffix=ext); os.close(fd)
        _cleanup.append(out)
        try:
            embed_to_file(fixture, out, model, WM_TEXT, DEVICE, audio_model)
            out_ext = Path(out).suffix.lower()
            if out_ext != ext:
                issues.append(f"{ext}: output has {out_ext}")
            elif not os.path.exists(out) or os.path.getsize(out) < 1024:
                issues.append(f"{ext}: output empty/missing")
            else:
                log.info(f"  {ext}: OK ({os.path.getsize(out)/1024:.0f}KB)")
        except Exception as e:
            issues.append(f"{ext}: {e}")
            log.warning(f"  {ext}: FAILED — {e}")

    if issues: log.warning(f"  Issues: {issues}")
    passed = len(issues) == 0
    _results["11_extension_preserved"] = (passed, f"{issues}" if issues else "all extensions OK")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 12: Attack — halve bitrate, watermark still detectable
# ═══════════════════════════════════════════════════════════════════════════════
def test_attack_bitrate_half():
    log.info("=" * 60)
    log.info("TEST 12: Attack — halve bitrate")
    log.info("=" * 60)
    wm_video = _embed_test_video()
    model, _ = _load_models()

    # Probe original bitrate
    info = _probe_media_info(wm_video)
    # Estimate bitrate from file size and duration
    dur = info["video"].get("duration", 0)
    if dur <= 0:
        dur = 70.0  # fallback
    orig_br = os.path.getsize(wm_video) * 8 / dur / 1000  # kbps
    half_br = int(orig_br / 2)
    log.info(f"  Original ~{orig_br:.0f}kbps → re-encode at {half_br}kbps")

    fd, attacked = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    _cleanup.append(attacked)
    ok = _ffmpeg(["-i", wm_video, "-c:v", "libx264", "-b:v", f"{half_br}k",
                  "-maxrate", f"{half_br}k", "-bufsize", f"{half_br*2}k",
                  "-c:a", "copy", attacked])
    if not ok:
        _results["12_attack_bitrate_half"] = (False, "ffmpeg re-encode failed")
        return False

    atk_sz = os.path.getsize(attacked) / 1024 / 1024
    log.info(f"  Attacked file: {atk_sz:.1f}MB")

    decode = _extract_video_watermark(attacked, model, max_frames=EXTRACT_FRAMES)
    detected = decode.get("correctable", False)
    extracted = decode.get("decoded_text", "")
    match = extracted == WM_TEXT
    log.info(f"  Detected={detected}, text='{extracted}', match={match}")
    passed = detected and match
    _results["12_attack_bitrate_half"] = (passed, f"br={half_br}k, detected={detected}, match={match}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 13: Attack — crop 10%
# ═══════════════════════════════════════════════════════════════════════════════
def test_attack_crop_10():
    log.info("=" * 60)
    log.info("TEST 13: Attack — crop 10% (remove 5% each side)")
    log.info("=" * 60)
    wm_video = _embed_test_video()
    model, _ = _load_models()

    info = _probe_media_info(wm_video)
    W = info["video"]["width"]
    H = info["video"]["height"]
    # Crop 5% from each edge → 90% of original
    cw = int(W * 0.9)
    ch = int(H * 0.9)
    # Make even
    cw = cw - cw % 2
    ch = ch - ch % 2
    cx = (W - cw) // 2
    cy = (H - ch) // 2
    log.info(f"  Original {W}x{H} → crop to {cw}x{ch} (offset {cx},{cy})")

    fd, attacked = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    _cleanup.append(attacked)
    ok = _ffmpeg(["-i", wm_video,
                  "-vf", f"crop={cw}:{ch}:{cx}:{cy}",
                  "-c:v", "libx264", "-crf", "18", "-c:a", "copy", attacked])
    if not ok:
        _results["13_attack_crop_10"] = (False, "ffmpeg crop failed")
        return False

    log.info(f"  Cropped file: {os.path.getsize(attacked)/1024/1024:.1f}MB")

    decode = _extract_video_watermark(attacked, model, max_frames=EXTRACT_FRAMES)
    detected = decode.get("correctable", False)
    extracted = decode.get("decoded_text", "")
    match = extracted == WM_TEXT
    log.info(f"  Detected={detected}, text='{extracted}', match={match}")
    passed = detected and match
    _results["13_attack_crop_10"] = (passed, f"crop={cw}x{ch}, detected={detected}, match={match}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 14: Attack — horizontal flip
# ═══════════════════════════════════════════════════════════════════════════════
def test_attack_hflip():
    log.info("=" * 60)
    log.info("TEST 14: Attack — horizontal flip")
    log.info("=" * 60)
    wm_video = _embed_test_video()
    model, _ = _load_models()

    fd, attacked = tempfile.mkstemp(suffix=".mp4"); os.close(fd)
    _cleanup.append(attacked)
    ok = _ffmpeg(["-i", wm_video, "-vf", "hflip",
                  "-c:v", "libx264", "-crf", "18", "-c:a", "copy", attacked])
    if not ok:
        _results["14_attack_hflip"] = (False, "ffmpeg hflip failed")
        return False

    log.info(f"  Flipped file: {os.path.getsize(attacked)/1024/1024:.1f}MB")

    decode = _extract_video_watermark(attacked, model, max_frames=EXTRACT_FRAMES)
    detected = decode.get("correctable", False)
    extracted = decode.get("decoded_text", "")
    match = extracted == WM_TEXT
    log.info(f"  Detected={detected}, text='{extracted}', match={match}")
    # Note: hflip is a strong attack — watermark survival depends on model training.
    # We log the result but don't necessarily expect it to pass.
    passed = detected and match
    _results["14_attack_hflip"] = (passed, f"detected={detected}, match={match}")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("  VIDEOSEAL PIPELINE TEST SUITE")
    log.info("=" * 70)
    log.info(f"  Input:  {INPUT_VIDEO}")
    log.info(f"  Model:  {CKPT}")
    log.info(f"  Text:   '{WM_TEXT}'")
    log.info(f"  Device: {DEVICE}")
    log.info("=" * 70)

    t0 = time.time()

    tests = [
        ("1. Video watermark",       test_video_watermark),
        ("2. Audio watermark",       test_audio_watermark),
        ("3. Video quality",         test_video_quality),
        ("4. File stability",        test_file_stability),
        ("5. No flicker",            test_no_flicker),
        ("6. Frame count",           test_frame_count),
        ("7. File size",             test_file_size),
        ("8. Params unchanged",      test_params_unchanged),
        ("9. MXF format",            test_mxf_format),
        ("10. WebM format",          test_webm_format),
        ("11. Extension preserved",  test_extension_preserved),
        ("12. Attack: bitrate/2",    test_attack_bitrate_half),
        ("13. Attack: crop 10%",     test_attack_crop_10),
        ("14. Attack: hflip",        test_attack_hflip),
    ]

    for name, fn in tests:
        try:
            log.info(f"\n>>> Running: {name}")
            fn()
        except Exception as e:
            log.error(f"  TEST CRASHED: {e}", exc_info=True)
            key = name.split(". ", 1)[1].lower().replace(" ", "_").replace(":", "").replace("/", "")
            _results[key] = (False, f"CRASH: {e}")

    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("  TEST RESULTS SUMMARY")
    log.info("=" * 70)
    n_pass = n_fail = 0
    for name, (passed, msg) in sorted(_results.items()):
        status = "PASS ✓" if passed else "FAIL ✗"
        if passed: n_pass += 1
        else: n_fail += 1
        log.info(f"  [{status}] {name}: {msg}")
    log.info("-" * 70)
    log.info(f"  Total: {n_pass + n_fail} tests, {n_pass} passed, {n_fail} failed  ({elapsed:.1f}s)")
    log.info("=" * 70)

    # Cleanup temp files
    for f in _cleanup:
        try:
            if os.path.exists(f): os.unlink(f)
        except Exception:
            pass

    torch.cuda.empty_cache()
    return n_fail == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
