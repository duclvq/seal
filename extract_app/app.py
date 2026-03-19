"""
VideoSeal Extract App — Simple web UI for watermark verification.
Extracts video + audio watermarks, looks up original embed info in DB.
"""

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# ── Project paths ──────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "audioseal" / "src"))
os.chdir(_PROJECT_ROOT)
os.environ.setdefault("NO_TORCH_COMPILE", "1")

import torch
import torchaudio
import torchaudio.functional as TAF
from videoseal.utils.cfg import setup_model_from_checkpoint
from web_demo.core.video_io import extract_audio_track
from web_demo.core.ecc import msg_tensor_to_text_bch

# ── Config ─────────────────────────────────────────────────────────────────────
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = _APP_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "900"))
CKPT_PATH = os.getenv("CKPT_PATH", "output/run2_video/checkpoint350.pth")
WATCHER_DB = os.getenv(
    "WATCHER_DB",
    str(_PROJECT_ROOT / "watcher_service" / "data" / "db" / "watermarks.db"),
)
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.70"))

# ── Globals ────────────────────────────────────────────────────────────────────
device = torch.device(DEVICE)
_video_model = None
_audio_detector = None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024 * 1024  # 5GB


@app.errorhandler(413)
def _err_413(e):
    return jsonify({"error": "File quá lớn. Hãy dùng đường dẫn server thay vì upload."}), 413


@app.errorhandler(500)
def _err_500(e):
    return jsonify({"error": f"Lỗi server: {e}"}), 500


def _get_video_model():
    global _video_model
    if _video_model is not None:
        return _video_model
    ckpt_path = Path(CKPT_PATH)
    if not ckpt_path.is_absolute():
        ckpt_path = _PROJECT_ROOT / ckpt_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _video_model = setup_model_from_checkpoint(str(ckpt_path))
    _video_model.eval().to(device)
    return _video_model


def _get_audio_detector():
    global _audio_detector
    if _audio_detector is not None:
        return _audio_detector
    try:
        # Force offline mode to avoid hanging on machines without internet
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import threading
        _det_result = [None]
        _det_error = [None]

        def _load_det():
            try:
                from audioseal import AudioSeal
                _det_result[0] = AudioSeal.load_detector("audioseal_detector_16bits").eval().to(device)
            except Exception as e:
                _det_error[0] = e

        t = threading.Thread(target=_load_det, daemon=True)
        t.start()
        t.join(timeout=15)

        if t.is_alive() or _det_error[0]:
            _audio_detector = False
        else:
            _audio_detector = _det_result[0]
    except Exception:
        _audio_detector = False
    return _audio_detector


# ── Audio helpers ─────────────────────────────────────────────────────────────
def _decode_audio_bits(bits: list) -> str:
    if len(bits) < 16:
        return ""
    chars = []
    for i in range(0, 16, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | int(bits[i + j])
        if 32 <= byte_val < 127:
            chars.append(chr(byte_val))
        else:
            chars.append("?")
    return "".join(chars)


def _encode_audio_key(text: str) -> list:
    txt2 = (text + "\x00\x00")[:2]
    bits = []
    for ch in txt2:
        b = ord(ch) & 0xFF
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits


# ── Audio extraction ──────────────────────────────────────────────────────────
def _extract_audio_watermark(video_path: str) -> dict:
    detector = _get_audio_detector()
    if detector is False:
        return {"detected": False, "audio_key": None}

    audio_tmp = extract_audio_track(video_path)
    if audio_tmp is None:
        return {"detected": False, "audio_key": None}

    try:
        wav, sr = torchaudio.load(audio_tmp)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 16000:
            wav = TAF.resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).to(device)

        with torch.no_grad():
            result, message = detector.detect_watermark(wav, 16000)

        detected = float(result) > 0.5
        audio_bits = message.squeeze().int().tolist() if message is not None else None
        audio_key = _decode_audio_bits(audio_bits) if audio_bits else None

        audio_match = None
        if detected and audio_bits:
            audio_match = _lookup_audio_in_db(audio_bits)

        return {
            "detected": bool(detected),
            "audio_key": audio_key,
            "audio_match": audio_match,
        }
    except Exception:
        return {"detected": False, "audio_key": None}
    finally:
        if audio_tmp and os.path.exists(audio_tmp):
            os.unlink(audio_tmp)


def _lookup_audio_in_db(extracted_bits: list) -> dict | None:
    if not os.path.exists(WATCHER_DB):
        return None
    try:
        conn = sqlite3.connect(WATCHER_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT DISTINCT watermark_text FROM jobs WHERE status='done' AND watermark_text IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        return None

    best_acc, best_text = 0.0, None
    for row in rows:
        wm_text = row["watermark_text"]
        ref_bits = _encode_audio_key(wm_text)
        if len(extracted_bits) != len(ref_bits):
            continue
        acc = sum(a == b for a, b in zip(extracted_bits, ref_bits)) / len(extracted_bits)
        if acc > best_acc:
            best_acc, best_text = acc, wm_text

    if best_text and best_acc >= 0.80:
        return {"watermark_text": best_text, "match": round(best_acc * 100, 1)}
    return None


# ── DB lookup ──────────────────────────────────────────────────────────────────
def _lookup_in_watcher_db(bits_list: list) -> dict | None:
    if not os.path.exists(WATCHER_DB):
        return None
    n = len(bits_list)
    if n == 0:
        return None

    try:
        conn = sqlite3.connect(WATCHER_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT filename, watermark_text, bits_list, created_at, input_path "
            "FROM jobs WHERE status='done' AND bits_list IS NOT NULL"
        ).fetchall()
        conn.close()
    except Exception:
        return None

    best, best_acc = None, 0.0
    for row in rows:
        try:
            stored = json.loads(row["bits_list"])
        except Exception:
            continue
        if len(stored) != n:
            continue
        acc = sum(a == b for a, b in zip(bits_list, stored)) / n
        if acc > best_acc:
            best_acc = acc
            best = dict(row)
            best["match_accuracy"] = round(acc * 100, 1)

    if best and best_acc >= MATCH_THRESHOLD:
        return best
    return None


# ── Streaming chunk extract helper ─────────────────────────────────────────────
def _extract_chunk(model, chunk_buf: list) -> torch.Tensor:
    """Extract watermark from a chunk of frames. Returns bit_preds [N, K] on CPU.
    Accepts list of uint8 [3, H, W] tensors — normalizes on GPU to save CPU RAM."""
    video_chunk = torch.stack(chunk_buf).to(device).float().div_(255.0)
    chunk_buf.clear()  # free CPU memory immediately
    with torch.no_grad():
        outputs = model.detect(video_chunk, is_video=True)
    preds = outputs["preds"]
    if preds.dim() == 4:
        preds = preds.mean(dim=(-2, -1))
    bit_preds = preds[:, 1:].cpu()
    del video_chunk, outputs, preds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bit_preds


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def api_extract():
    video_path = None
    tmp_file = None

    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        tmp_file = UPLOAD_FOLDER / f.filename
        f.save(str(tmp_file))
        video_path = str(tmp_file)
    elif request.form.get("path"):
        video_path = request.form["path"]
    else:
        return jsonify({"error": "Vui lòng chọn file video hoặc nhập đường dẫn"}), 400

    if not os.path.isfile(video_path):
        return jsonify({"error": f"File không tồn tại: {video_path}"}), 400

    try:
        t0 = time.time()
        model = _get_video_model()

        # ── Streaming extract: decode + detect chunk-by-chunk ──
        # Keeps frames as uint8 on CPU (4× less RAM than float32).
        # Running aggregation — never accumulates all preds in memory.
        import av
        CHUNK_SIZE = min(int(os.getenv("EXTRACT_CHUNK", "8")), 16)  # small chunks to limit VRAM

        ext = os.path.splitext(video_path)[1].lstrip(".").lower()
        from web_demo.core.video_io import _FFMPEG_CONVERT_EXT, _to_mp4_via_ffmpeg
        tmp_converted = None
        av_path = video_path
        if ext in _FFMPEG_CONVERT_EXT:
            tmp_converted = _to_mp4_via_ffmpeg(video_path)
            av_path = tmp_converted

        # Running aggregation: accumulate sum of (bit_preds * |bit_preds|) and count
        # instead of storing all preds in memory
        running_sum = None  # [K] tensor, accumulated on CPU
        n_frames = 0
        fps = 24.0
        chunk_buf = []

        try:
            with av.open(av_path) as container:
                stream = container.streams.video[0]
                rate = stream.average_rate or stream.guessed_rate
                if rate:
                    fps = float(rate)
                stream.thread_type = "AUTO"

                for frame in container.decode(stream):
                    # Keep as uint8 on CPU — 4× less RAM than float32
                    rgb = frame.to_ndarray(format="rgb24")
                    t = torch.from_numpy(rgb).permute(2, 0, 1)  # uint8 [3, H, W]
                    chunk_buf.append(t)

                    if len(chunk_buf) >= CHUNK_SIZE:
                        preds = _extract_chunk(model, chunk_buf)  # [N, K] cpu; clears chunk_buf
                        # Running aggregation: squared_avg
                        chunk_sum = (preds * preds.abs()).sum(dim=0)  # [K]
                        if running_sum is None:
                            running_sum = chunk_sum
                        else:
                            running_sum.add_(chunk_sum)
                        n_frames += preds.shape[0]
                        del preds, chunk_sum

                    if n_frames + len(chunk_buf) >= MAX_FRAMES:
                        break

                # Flush remaining frames
                if chunk_buf:
                    preds = _extract_chunk(model, chunk_buf)
                    chunk_sum = (preds * preds.abs()).sum(dim=0)
                    if running_sum is None:
                        running_sum = chunk_sum
                    else:
                        running_sum.add_(chunk_sum)
                    n_frames += preds.shape[0]
                    del preds, chunk_sum
        finally:
            if tmp_converted and os.path.exists(tmp_converted):
                os.unlink(tmp_converted)

        if running_sum is None or n_frames == 0:
            return jsonify({"error": "Không đọc được frame nào từ video"}), 400

        # Final aggregation
        decoded_msg = running_sum / n_frames  # [K]
        msg = (decoded_msg > 0).unsqueeze(0)  # [1, K]
        del running_sum

        decode = msg_tensor_to_text_bch(msg)
        video_detected = decode["correctable"]
        db_match = _lookup_in_watcher_db(decode["bits_list"])

        audio_result = _extract_audio_watermark(video_path)
        elapsed = time.time() - t0

        result = {
            "video_watermark": video_detected,
            "video_key": decode["decoded_text"] or "",
            "audio_watermark": audio_result["detected"],
            "audio_key": audio_result.get("audio_key") or "",
            "frames_analyzed": n_frames,
            "extract_time": round(elapsed, 1),
            "original_filename": db_match.get("filename", "") if db_match else None,
            "embed_time": db_match.get("created_at", "") if db_match else None,
            "audio_db_match": (audio_result.get("audio_match") or {}).get("watermark_text"),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_file and tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    print(f"Device: {device}")
    print(f"Model: {CKPT_PATH}")
    print(f"Watcher DB: {WATCHER_DB}")
    print(f"Match threshold: {MATCH_THRESHOLD}")
    app.run(host="0.0.0.0", port=5001, debug=False)
