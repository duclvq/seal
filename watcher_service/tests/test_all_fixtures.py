"""
Full pipeline test: embed → extract watermark on ALL fixture formats.
Fixtures: mp4, mkv, mov, mxf, webm

Usage:
    python watcher_service/tests/test_all_fixtures.py
"""
import logging
import os
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
os.environ.setdefault("FFMPEG_DECODE", "true")
os.chdir(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

import av
import torch
from worker import load_video_model, load_audio_model, embed_to_file
from web_demo.core.ecc import msg_tensor_to_text_bch

CKPT     = "output/run2_video/checkpoint350.pth"
FIXTURES = _TESTS_DIR / "fixtures"
WM_TEXT  = "test"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTRACT_CHUNK  = 8
EXTRACT_FRAMES = 300

_model = None
_audio_model = None
_cleanup = []


def _load_models():
    global _model, _audio_model
    if _model is None:
        log.info("Loading video model...")
        _model = load_video_model(CKPT, DEVICE)
        log.info("Loading audio model...")
        _audio_model = load_audio_model(DEVICE)
    return _model, _audio_model


def _extract_wm(video_path: str, model, max_frames: int = EXTRACT_FRAMES) -> dict:
    """Extract video watermark via streaming aggregation."""
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
                batch = torch.stack(chunk_buf).to(DEVICE).float().div_(255.0)
                chunk_buf.clear()
                with torch.no_grad():
                    out = model.detect(batch, is_video=True)
                preds = out["preds"]
                if preds.dim() == 4:
                    preds = preds.mean(dim=(-2, -1))
                bp = preds[:, 1:].cpu()
                cs = (bp * bp.abs()).sum(dim=0)
                running_sum = cs if running_sum is None else running_sum + cs
                n_frames += bp.shape[0]
                del batch, out, preds, bp, cs
                torch.cuda.empty_cache()
            if n_frames >= max_frames:
                break
        if chunk_buf and n_frames < max_frames:
            batch = torch.stack(chunk_buf).to(DEVICE).float().div_(255.0)
            chunk_buf.clear()
            with torch.no_grad():
                out = model.detect(batch, is_video=True)
            preds = out["preds"]
            if preds.dim() == 4:
                preds = preds.mean(dim=(-2, -1))
            bp = preds[:, 1:].cpu()
            cs = (bp * bp.abs()).sum(dim=0)
            running_sum = cs if running_sum is None else running_sum + cs
            n_frames += bp.shape[0]
            del batch, out, preds, bp, cs

    if running_sum is None:
        return {"correctable": False, "decoded_text": None, "n_frames_used": 0}

    msg = (running_sum / n_frames > 0).unsqueeze(0)
    decode = msg_tensor_to_text_bch(msg)
    decode["n_frames_used"] = n_frames
    return decode


def _run_one(fixture_path: str, out_ext: str) -> dict:
    """Embed → extract on one fixture. Returns result dict."""
    model, audio_model = _load_models()
    fd, out_path = tempfile.mkstemp(suffix=f".{out_ext}")
    os.close(fd)
    _cleanup.append(out_path)

    t0 = time.time()
    result = embed_to_file(
        input_path=fixture_path,
        output_path=out_path,
        video_model=model,
        watermark_text=WM_TEXT,
        device=DEVICE,
        audio_model=audio_model,
    )
    t_embed = time.time() - t0

    # Verify output exists
    out_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    # Extract watermark
    t0 = time.time()
    decode = _extract_wm(out_path, model)
    t_extract = time.time() - t0

    return {
        "embed_time": t_embed,
        "extract_time": t_extract,
        "total_frames": result.get("total_frames", 0),
        "mode": result.get("mode", "?"),
        "out_size_mb": out_size / 1024 / 1024,
        "detected": decode.get("correctable", False),
        "extracted_text": decode.get("decoded_text", ""),
        "match": decode.get("decoded_text", "") == WM_TEXT,
        "n_frames_extract": decode.get("n_frames_used", 0),
    }


def main():
    fixtures = sorted(FIXTURES.glob("test_clip.*"))
    if not fixtures:
        log.error(f"No fixtures found in {FIXTURES}")
        return False

    log.info(f"Found {len(fixtures)} fixtures: {[f.suffix for f in fixtures]}")
    log.info(f"Watermark text: '{WM_TEXT}', device: {DEVICE}")
    log.info("=" * 70)

    results = {}
    for fx in fixtures:
        ext = fx.suffix.lstrip(".")
        name = fx.name
        log.info(f"\n{'─'*70}")
        log.info(f"[{ext.upper()}] {name}")
        log.info(f"{'─'*70}")
        try:
            r = _run_one(str(fx), ext)
            results[ext] = r
            status = "PASS" if r["match"] else "FAIL"
            log.info(
                f"  [{status}] detected={r['detected']}, text='{r['extracted_text']}', "
                f"match={r['match']}, frames={r['total_frames']}, "
                f"embed={r['embed_time']:.1f}s, extract={r['extract_time']:.1f}s, "
                f"out={r['out_size_mb']:.1f}MB, mode={r['mode']}"
            )
        except Exception as e:
            log.error(f"  [ERROR] {e}", exc_info=True)
            results[ext] = {"match": False, "error": str(e)}

    # Cleanup temp files
    for p in _cleanup:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass

    # Summary
    print(f"\n{'='*70}")
    print(f"  EMBED → EXTRACT RESULTS  (text='{WM_TEXT}')")
    print(f"{'='*70}")
    print(f"  {'Format':<8} {'Status':<8} {'Detected':<10} {'Text':<10} {'Frames':<8} {'Embed':<8} {'Extract':<8} {'Size':<8}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for ext, r in results.items():
        if "error" in r:
            print(f"  {ext:<8} {'ERROR':<8} {'':<10} {'':<10} {'':<8} {'':<8} {'':<8} {r['error'][:30]}")
        else:
            icon = "✓" if r["match"] else "✗"
            print(
                f"  {ext:<8} {icon+' PASS' if r['match'] else icon+' FAIL':<8} "
                f"{str(r['detected']):<10} {repr(r['extracted_text']):<10} "
                f"{r['total_frames']:<8} {r['embed_time']:<8.1f} {r['extract_time']:<8.1f} "
                f"{r['out_size_mb']:<8.1f}"
            )

    n_pass = sum(1 for r in results.values() if r.get("match"))
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} passed")
    print(f"{'='*70}")
    return n_pass == n_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
