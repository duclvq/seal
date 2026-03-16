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
from web_demo.core.video_io import load_video_tensor, extract_audio_track
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
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB


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
        from audioseal import AudioSeal
        _audio_detector = AudioSeal.load_detector("audioseal_detector_16bits").eval().to(device)
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
        video, fps = load_video_tensor(video_path, max_frames=MAX_FRAMES)
        n_frames = video.shape[0]

        with torch.no_grad():
            msg = model.extract_message(
                video.to(device), aggregation="squared_avg"
            )

        decode = msg_tensor_to_text_bch(msg.cpu())
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
