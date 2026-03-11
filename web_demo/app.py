"""
VideoSeal Web Demo — Flask Application
Run from d:\py_source\videoseal\web_demo\:
    python app.py
Then open http://localhost:5000
"""

import os
import sys
import time

import torch
from flask import Flask, request, jsonify, render_template, send_file, abort
from werkzeug.utils import secure_filename

# Change CWD to the videoseal project root so that relative config paths
# (e.g. "configs/attenuation.yaml") resolve correctly, regardless of where
# this script is launched from.
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(_PROJECT_ROOT)

# Make sure the parent videoseal package is importable
sys.path.insert(0, _PROJECT_ROOT)

from config import (
    UPLOAD_FOLDER, SESSION_FOLDER, ALLOWED_EXT,
    ALLOWED_AUDIO_EXT,
    MAX_CONTENT_MB, DEVICE, DEFAULT_K, RS_TOTAL_BYTES, MSG_BITS,
)
from core.model_manager import get_model, get_embed_model, get_extract_model, model_mode_label
from core.ecc            import (
    text_to_msg_tensor, msg_tensor_to_text, rs_info,
    text_to_msg_tensor_bch, msg_tensor_to_text_bch, bch_info, BCH_AVAILABLE,
    text_to_msg_tensor_ldpc, msg_tensor_to_text_ldpc, ldpc_info,
)
from core.video_io       import (
    load_video_tensor, save_video_tensor,
    save_session_meta, load_session_meta,
    session_video_path, find_session_video, new_session_id,
    extract_audio_track, embed_video_streaming, save_original_copy,
    crop_center_for_extract,
    _EXT_TO_FORMAT, _EXT_MIME,
)
from core.attacks           import ATTACK_REGISTRY, ATTACK_DISPLAY_NAMES
from core.audio_watermark   import (
    embed_audio, detect_audio, run_audio_attack, AUDIO_ATTACKS,
    get_models as get_audio_models,
)
from core.db       import init_db, insert_watermark, find_nearest
from core.temporal import detect_temporal

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(SESSION_FOLDER, exist_ok=True)

# Initialise hidden watermark registry
init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def _allowed_audio(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXT

def _whitelist_video_name(name: str) -> bool:
    """Prevent path traversal by whitelisting allowed video names."""
    if name in ("original", "watermarked", "attacked_upload", "attacked_meta_overwrite"):
        return True
    for key in ATTACK_REGISTRY:
        if name == f"attacked_{key}":
            return True
    return False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/encode", methods=["POST"])
def api_encode():
    """
    Multipart form:
        video    : file
        text     : str
        ecc_type : "rs" | "bch"   (default "rs")
        k        : int  (RS data bytes,  used when ecc_type="rs")
        t        : int  (BCH correction capability, used when ecc_type="bch")

    Returns JSON with session_id, bits_list, video URLs, embed_time, etc.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    f = request.files["video"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": f"Unsupported video format. Supported: {', '.join(sorted(ALLOWED_EXT)).upper()}"}), 400

    text        = request.form.get("text", "VideoSeal Demo")
    ecc_type    = request.form.get("ecc_type", "rs").strip().lower()
    model_mode  = request.form.get("model_mode", "meta").strip().lower()
    center_mask = request.form.get("center_mask", "false").strip().lower() == "true"
    if ecc_type not in ("rs", "bch", "ldpc"):
        ecc_type = "rs"
    if model_mode not in ("meta", "custom", "hybrid"):
        model_mode = "meta"

    # Parse params and encode text → MSG_BITS-bit tensor
    if ecc_type == "bch":
        if not BCH_AVAILABLE:
            return jsonify({"error": "bchlib not installed on server. Run: pip install bchlib"}), 500
        try:
            msg_tensor, codeword, bits_list = text_to_msg_tensor_bch(text, msg_bits=MSG_BITS)
        except (ValueError, ImportError) as e:
            return jsonify({"error": str(e)}), 400
        k = DEFAULT_K   # stored for legacy compat
        info = bch_info(MSG_BITS)
        extra = {"ecc_type": "bch",
                 "ecc_bytes": info["ecc_bytes"], "max_errors": info["max_bit_errors"]}
    elif ecc_type == "ldpc":
        try:
            msg_tensor, codeword, bits_list = text_to_msg_tensor_ldpc(text, n_bits=MSG_BITS)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        k = DEFAULT_K   # stored for legacy compat
        info = ldpc_info(MSG_BITS)
        extra = {"ecc_type": "ldpc",
                 "max_text_bytes": info["max_text_bytes"],
                 "rate": info["rate"],
                 "d_v": info["d_v"]}
    else:
        try:
            k = int(request.form.get("k", DEFAULT_K))
        except ValueError:
            return jsonify({"error": "k must be an integer"}), 400
        if not (1 <= k <= RS_TOTAL_BYTES - 2):
            return jsonify({"error": f"k must be between 1 and {RS_TOTAL_BYTES - 2}"}), 400
        try:
            msg_tensor, codeword, bits_list = text_to_msg_tensor(text, k, RS_TOTAL_BYTES)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        info = rs_info(k, RS_TOTAL_BYTES)
        extra = {"k": k, "ecc_type": "rs",
                 "ecc_bytes": info["ecc_bytes"], "max_errors": info["max_byte_errors"]}

    # Save uploaded file to a temp location
    session_id = new_session_id()
    safe_name  = secure_filename(f.filename)
    tmp_path   = os.path.join(UPLOAD_FOLDER, f"{session_id}_{safe_name}")
    f.save(tmp_path)

    # Determine output container format (match input extension)
    input_ext = os.path.splitext(safe_name)[1].lstrip(".").lower() or "mp4"
    out_ext   = input_ext if input_ext in _EXT_TO_FORMAT else "mp4"

    audio_tmp = None
    try:
        # Extract audio track from original before processing
        audio_tmp = extract_audio_track(tmp_path)

        embed_model = get_embed_model(model_mode)

        # Save original (ffmpeg re-encode, no frame limit)
        save_original_copy(tmp_path, session_id, "original",
                           audio_src=audio_tmp, out_ext=out_ext)

        # Embed watermark — chunk-based, full video length
        t0 = time.time()
        _, fps, num_frames = embed_video_streaming(
            tmp_path, session_id, "watermarked",
            embed_model, msg_tensor, DEVICE,
            audio_src=audio_tmp, out_ext=out_ext,
            center_mask=center_mask,
        )
        embed_time = round(time.time() - t0, 2)

        # Persist metadata for attack phase (include model_mode + center_mask for extraction)
        save_session_meta(session_id, bits_list, k, fps, ecc_type=ecc_type,
                          out_ext=out_ext, model_mode=model_mode,
                          center_mask=center_mask)

        # Log to hidden watermark registry
        insert_watermark(session_id, bits_list, text, ecc_type, k, safe_name)

    finally:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        os.remove(tmp_path)
        if audio_tmp and os.path.exists(audio_tmp):
            os.unlink(audio_tmp)

    return jsonify({
        "session_id":      session_id,
        "bits_list":       bits_list,
        "codeword_hex":    codeword.hex(),
        "text":            text,
        "fps":             fps,
        "num_frames":      num_frames,
        "original_url":    f"/api/video/{session_id}/original",
        "watermarked_url": f"/api/video/{session_id}/watermarked",
        "out_ext":         out_ext,
        "embed_time_s":    embed_time,
        "model_mode":      model_mode,
        "model_mode_label": model_mode_label(model_mode),
        **extra,
    })


@app.route("/api/video/<session_id>/<name>")
def serve_video(session_id, name):
    """Stream a video from the session folder (whitelisted names only)."""
    if not _whitelist_video_name(name):
        abort(404)
    path = find_session_video(session_id, name)
    if not path or not os.path.exists(path):
        abort(404)
    ext  = os.path.splitext(path)[1].lstrip(".").lower()
    mime = _EXT_MIME.get(ext, "video/mp4")
    dl   = request.args.get("dl") == "1"
    dl_name = request.args.get("filename") or f"{name}.{ext}"
    return send_file(
        path, mimetype=mime,
        as_attachment=dl,
        download_name=dl_name if dl else None,
        conditional=not dl,
    )


@app.route("/api/attacks/run", methods=["POST"])
def api_attacks_run():
    """
    JSON body:
        session_id : str
        attacks    : list[str] | "all"
        k          : int

    Returns JSON with results array.
    """
    data       = request.get_json(force=True)
    session_id = data.get("session_id", "")
    attacks    = data.get("attacks", "all")
    ecc_type   = data.get("ecc_type", "rs").strip().lower()
    if ecc_type not in ("rs", "bch", "ldpc"):
        ecc_type = "rs"
    try:
        k = int(data.get("k", DEFAULT_K))
    except (ValueError, TypeError):
        k = DEFAULT_K

    if attacks == "all" or attacks == ["all"]:
        attacks = list(ATTACK_REGISTRY.keys())

    # Validate session
    wm_path = find_session_video(session_id, "watermarked")
    if not wm_path or not os.path.exists(wm_path):
        return jsonify({"error": "Session not found or video not yet embedded."}), 404

    meta          = load_session_meta(session_id)
    original_bits = meta.get("bits_list")
    fps           = meta.get("fps", 24.0)
    # Prefer ecc params from session meta (they were set at embed time)
    ecc_type    = meta.get("ecc_type", ecc_type)
    k           = meta.get("k", k)
    model_mode  = meta.get("model_mode", "meta")
    center_mask = meta.get("center_mask", False)

    # Load watermarked video (CPU) → move to GPU
    video_w_cpu, _ = load_video_tensor(wm_path)
    video_w = video_w_cpu.to(DEVICE)

    model = get_extract_model(model_mode)
    results = []

    for attack_name in attacks:
        if attack_name not in ATTACK_REGISTRY:
            continue

        attack_fn = ATTACK_REGISTRY[attack_name]
        t0 = time.time()

        try:
            with torch.no_grad():
                attacked = attack_fn(video_w, fps=fps)
                attacked_gpu = attacked.to(DEVICE)
                extract_input = crop_center_for_extract(attacked_gpu) if center_mask else attacked_gpu
                extracted_msg = model.extract_message(extract_input, aggregation="avg")
                extracted_msg_cpu = extracted_msg.cpu()
            del attacked_gpu, extracted_msg
        except Exception as e:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            results.append({
                "attack":           attack_name,
                "display_name":     ATTACK_DISPLAY_NAMES.get(attack_name, attack_name),
                "error":            str(e),
                "bit_accuracy":     0.0,
                "pass":             False,
                "decoded_text":     None,
                "correctable":      False,
                "errors_corrected": -1,
                "bits_list":        [],
                "attacked_url":     "",
                "process_time_s":   round(time.time() - t0, 2),
            })
            continue

        process_time = round(time.time() - t0, 2)

        # Save attacked video to session folder (CPU)
        save_video_tensor(attacked.cpu(), fps, session_id, f"attacked_{attack_name}")
        del attacked

        # Bit accuracy vs original
        extracted_bits = [int(b) for b in extracted_msg_cpu.squeeze(0).tolist()]
        if original_bits and len(original_bits) == len(extracted_bits):
            matches = sum(a == b for a, b in zip(original_bits, extracted_bits))
            bit_acc = matches / len(original_bits)
        else:
            bit_acc = 0.0

        # Decode text with ECC
        if ecc_type == "bch":
            decode_result = msg_tensor_to_text_bch(extracted_msg_cpu)
        elif ecc_type == "ldpc":
            decode_result = msg_tensor_to_text_ldpc(extracted_msg_cpu)
        else:
            decode_result = msg_tensor_to_text(extracted_msg_cpu, k, RS_TOTAL_BYTES)

        db_match = find_nearest(decode_result["bits_list"])
        results.append({
            "attack":           attack_name,
            "display_name":     ATTACK_DISPLAY_NAMES.get(attack_name, attack_name),
            "bit_accuracy":     round(bit_acc, 4),
            "pass":             bit_acc >= 0.70,
            "decoded_text":     decode_result["decoded_text"],
            "correctable":      decode_result["correctable"],
            "errors_corrected": decode_result["errors_corrected"],
            "bits_list":        decode_result["bits_list"],
            "attacked_url":     f"/api/video/{session_id}/attacked_{attack_name}",
            "process_time_s":   process_time,
            "db_match":         db_match,
        })

    del video_w, video_w_cpu
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return jsonify({"results": results})


@app.route("/api/attacks/upload_video", methods=["POST"])
def api_attacks_upload_video():
    """
    Upload an externally captured/attacked video and extract its watermark.

    Multipart form:
        session_id   : str
        video        : file
        attack_label : str  (display name, optional)

    Returns the same JSON shape as a single entry in /api/attacks/run results.
    """
    session_id   = request.form.get("session_id", "")
    attack_label = request.form.get("attack_label", "Upload video thực")

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    f = request.files["video"]
    if not f.filename or not _allowed(f.filename):
        return jsonify({"error": f"Unsupported video format. Supported: {', '.join(sorted(ALLOWED_EXT)).upper()}"}), 400

    wm_path = find_session_video(session_id, "watermarked")
    if not wm_path or not os.path.exists(wm_path):
        return jsonify({"error": "Session not found or video not yet embedded."}), 404

    meta          = load_session_meta(session_id)
    original_bits = meta.get("bits_list")
    fps           = meta.get("fps", 24.0)
    ecc_type      = meta.get("ecc_type", "rs")
    k             = meta.get("k", DEFAULT_K)

    safe_name = secure_filename(f.filename)
    tmp_path  = os.path.join(UPLOAD_FOLDER, f"{session_id}_upload_{safe_name}")
    f.save(tmp_path)

    try:
        t0 = time.time()
        video_cpu, _ = load_video_tensor(tmp_path)
        video_gpu     = video_cpu.to(DEVICE)
        model_mode    = meta.get("model_mode", "meta")
        center_mask   = meta.get("center_mask", False)
        model         = get_extract_model(model_mode)

        with torch.no_grad():
            extract_input = crop_center_for_extract(video_gpu) if center_mask else video_gpu
            extracted_msg = model.extract_message(extract_input, aggregation="avg")
        extracted_msg_cpu = extracted_msg.cpu()
        del video_gpu, extracted_msg

        process_time = round(time.time() - t0, 2)

        # Save uploaded video under session folder for playback
        save_video_tensor(video_cpu, fps, session_id, "attacked_upload")

        extracted_bits = [int(b) for b in extracted_msg_cpu.squeeze(0).tolist()]
        if original_bits and len(original_bits) == len(extracted_bits):
            matches = sum(a == b for a, b in zip(original_bits, extracted_bits))
            bit_acc = matches / len(original_bits)
        else:
            bit_acc = 0.0

        if ecc_type == "bch":
            decode_result = msg_tensor_to_text_bch(extracted_msg_cpu)
        elif ecc_type == "ldpc":
            decode_result = msg_tensor_to_text_ldpc(extracted_msg_cpu)
        else:
            decode_result = msg_tensor_to_text(extracted_msg_cpu, k, RS_TOTAL_BYTES)

    finally:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        os.remove(tmp_path)

    db_match = find_nearest(decode_result["bits_list"])
    return jsonify({
        "attack":           "upload",
        "display_name":     attack_label,
        "bit_accuracy":     round(bit_acc, 4),
        "pass":             bit_acc >= 0.70,
        "decoded_text":     decode_result["decoded_text"],
        "correctable":      decode_result["correctable"],
        "errors_corrected": decode_result["errors_corrected"],
        "bits_list":        decode_result["bits_list"],
        "attacked_url":     f"/api/video/{session_id}/attacked_upload",
        "process_time_s":   process_time,
        "db_match":         db_match,
    })


@app.route("/api/attacks/meta_overwrite", methods=["POST"])
def api_attacks_meta_overwrite():
    """
    Re-watermark attack simulation:
      1. Take the session's custom-embedded watermarked video.
      2. Re-embed a RANDOM message using Meta's public model (attacker simulation).
      3. Extract with the custom model  → did original watermark survive?
      4. Extract with Meta's model      → did the overwrite succeed?

    JSON body: { "session_id": "..." }

    Returns:
      process_time_s, attacked_url,
      custom_extract: { bit_accuracy, pass, decoded_text, correctable, bits_list } | { error }
      meta_extract:   { bit_accuracy, pass, bits_list }
      original_model_mode: the mode used at embed time
    """
    data       = request.get_json(force=True)
    session_id = data.get("session_id", "")

    wm_path = find_session_video(session_id, "watermarked")
    if not wm_path or not os.path.exists(wm_path):
        return jsonify({"error": "Session not found or video not yet embedded."}), 404

    meta          = load_session_meta(session_id)
    original_bits = meta.get("bits_list")
    fps           = meta.get("fps", 24.0)
    ecc_type      = meta.get("ecc_type", "rs")
    k             = meta.get("k", DEFAULT_K)
    model_mode    = meta.get("model_mode", "meta")
    center_mask   = meta.get("center_mask", False)

    # Random overwrite message (the "attacker" embeds random bits)
    overwrite_msg  = torch.randint(0, 2, (1, MSG_BITS), dtype=torch.float32)
    overwrite_bits = [int(b) for b in overwrite_msg.squeeze(0).tolist()]

    custom_result    = None
    bit_acc_meta     = 0.0
    extracted_meta_bits = []
    process_time     = 0.0

    try:
        video_cpu, _ = load_video_tensor(wm_path)
        video_gpu    = video_cpu.to(DEVICE)

        t0 = time.time()

        # Step 1: Re-embed with Meta model (attacker overwrites)
        meta_embed = get_embed_model("meta")
        with torch.no_grad():
            outputs    = meta_embed.embed(video_gpu, msgs=overwrite_msg.to(DEVICE), is_video=True)
        rewater_gpu = outputs["imgs_w"]
        del outputs, video_gpu

        # Step 2: Extract with custom model → did our watermark survive?
        try:
            custom_extract_model = get_extract_model("custom")
            with torch.no_grad():
                extract_input_c = crop_center_for_extract(rewater_gpu) if center_mask else rewater_gpu
                extracted_c     = custom_extract_model.extract_message(extract_input_c, aggregation="avg")
            extracted_c_cpu = extracted_c.cpu()
            del extracted_c

            c_bits = [int(b) for b in extracted_c_cpu.squeeze(0).tolist()]
            if original_bits and len(original_bits) == len(c_bits):
                bit_acc_c = sum(a == b for a, b in zip(original_bits, c_bits)) / len(original_bits)
            else:
                bit_acc_c = 0.0

            if ecc_type == "bch":
                dec_c = msg_tensor_to_text_bch(extracted_c_cpu)
            elif ecc_type == "ldpc":
                dec_c = msg_tensor_to_text_ldpc(extracted_c_cpu)
            else:
                dec_c = msg_tensor_to_text(extracted_c_cpu, k, RS_TOTAL_BYTES)

            custom_result = {
                "bit_accuracy": round(bit_acc_c, 4),
                "pass":         bit_acc_c >= 0.70,
                "decoded_text": dec_c["decoded_text"],
                "correctable":  dec_c["correctable"],
                "bits_list":    c_bits,
            }
        except FileNotFoundError as e:
            custom_result = {"error": str(e)}

        # Step 3: Extract with Meta model → did the overwrite succeed?
        meta_extract_model = get_extract_model("meta")
        with torch.no_grad():
            extracted_m     = meta_extract_model.extract_message(rewater_gpu, aggregation="avg")
        extracted_m_cpu = extracted_m.cpu()
        del extracted_m

        extracted_meta_bits = [int(b) for b in extracted_m_cpu.squeeze(0).tolist()]
        bit_acc_meta = (
            sum(a == b for a, b in zip(overwrite_bits, extracted_meta_bits)) / len(overwrite_bits)
        )

        process_time = round(time.time() - t0, 2)

        # Save overwritten video for playback
        save_video_tensor(rewater_gpu.cpu(), fps, session_id, "attacked_meta_overwrite")
        del rewater_gpu

    finally:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return jsonify({
        "process_time_s":      process_time,
        "attacked_url":        f"/api/video/{session_id}/attacked_meta_overwrite",
        "custom_extract":      custom_result,
        "meta_extract": {
            "bit_accuracy": round(bit_acc_meta, 4),
            "pass":         bit_acc_meta >= 0.70,
            "bits_list":    extracted_meta_bits,
        },
        "overwrite_bits":      overwrite_bits,
        "original_model_mode": model_mode,
    })


@app.route("/api/video/detect_temporal", methods=["POST"])
def api_detect_temporal():
    """
    Temporal watermark analysis — splits video into 1-second windows.

    Two modes:
    A) Session mode (JSON body):   { "session_id": "..." }
       Uses the session's watermarked.mp4 as the video to analyse.
    B) Upload mode (multipart):    video file + optional session_id for reference bits.
       Analyses an uploaded (possibly spliced) video.

    Returns JSON:
        { segments, total_duration_s, window_s, fps }
    """
    import json as _json

    session_id = None
    video_path = None
    tmp_path   = None
    reference_bits = None
    ecc_type   = "bch"   # default for upload mode (4 data bytes)
    k          = DEFAULT_K

    # ── Determine mode ────────────────────────────────────────────────────
    if request.content_type and "multipart" in request.content_type:
        # Upload mode
        session_id   = request.form.get("session_id", "")
        f            = request.files.get("video")
        if f is None or not f.filename or not _allowed(f.filename):
            return jsonify({"error": "No valid video file provided"}), 400

        safe_name = secure_filename(f.filename)
        tmp_path  = os.path.join(UPLOAD_FOLDER, f"tmp_temporal_{safe_name}")
        f.save(tmp_path)
        video_path = tmp_path
    else:
        # Session mode (JSON)
        data       = request.get_json(force=True) or {}
        session_id = data.get("session_id", "")
        wm_path    = find_session_video(session_id, "watermarked")
        if not wm_path or not os.path.exists(wm_path):
            return jsonify({"error": "Session not found."}), 404
        video_path = wm_path

    # Load reference bits from session if available
    center_mask = False
    if session_id:
        meta = load_session_meta(session_id)
        reference_bits = meta.get("bits_list")
        ecc_type       = meta.get("ecc_type", "rs")
        k              = meta.get("k", DEFAULT_K)
        center_mask    = meta.get("center_mask", False)

    # Build ECC decoder callable
    def _decode(msg_tensor):
        if ecc_type == "bch":
            return msg_tensor_to_text_bch(msg_tensor)
        elif ecc_type == "ldpc":
            return msg_tensor_to_text_ldpc(msg_tensor)
        else:
            return msg_tensor_to_text(msg_tensor, k, RS_TOTAL_BYTES)

    try:
        model  = get_model()
        result = detect_temporal(
            video_path    = video_path,
            model         = model,
            device        = DEVICE,
            ecc_decoder   = _decode,
            reference_bits= reference_bits,
            db_lookup     = find_nearest,
            center_mask   = center_mask,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return jsonify(result)


@app.route("/api/audio/detect_temporal", methods=["POST"])
def api_audio_detect_temporal():
    """
    Temporal watermark detection for audio — splits into 1-second windows.
    Upload mode only: multipart with field "audio".
    Returns { segments, total_duration_s, window_s, sample_rate }
    Each segment: { start_s, end_s, detected, detection_prob, decoded_text }
    """
    from core.audio_watermark import get_models, msg_to_text, SAMPLE_RATE
    import torchaudio
    import torchaudio.functional as TAF

    f = request.files.get("audio")
    if f is None or not f.filename:
        return jsonify({"error": "No audio file provided"}), 400
    if not _allowed_audio(f.filename):
        return jsonify({"error": f"Unsupported audio format. Supported: {', '.join(sorted(ALLOWED_AUDIO_EXT)).upper()}"}), 400

    safe_name = secure_filename(f.filename)
    tmp_path  = os.path.join(UPLOAD_FOLDER, f"tmp_audio_temporal_{safe_name}")
    f.save(tmp_path)

    try:
        wav, sr = torchaudio.load(tmp_path)
        if sr != SAMPLE_RATE:
            wav = TAF.resample(wav, sr, SAMPLE_RATE)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # stereo → mono
        wav = wav.float().to(DEVICE)

        _, detector = get_models(DEVICE)

        total_samples = wav.shape[-1]
        window_samp   = SAMPLE_RATE          # 1 second = 16 000 samples
        total_dur     = total_samples / SAMPLE_RATE

        segments = []
        start = 0
        while start < total_samples:
            end   = min(start + window_samp, total_samples)
            chunk = wav[..., start:end]

            # Pad last chunk to full window length
            if chunk.shape[-1] < window_samp:
                chunk = torch.nn.functional.pad(chunk, (0, window_samp - chunk.shape[-1]))

            with torch.no_grad():
                det_prob_t, msg_t = detector.detect_watermark(chunk.unsqueeze(0))

            det_prob = float(det_prob_t[0])
            decoded  = msg_to_text(msg_t.float()) if det_prob >= 0.5 else None
            bits     = [int(b) for b in msg_t.float().squeeze(0).tolist()] if det_prob >= 0.5 else []
            db_match = find_nearest(bits) if bits else None
            detected = det_prob >= 0.5 or bool(db_match)

            segments.append({
                "start_s":        round(start / SAMPLE_RATE, 3),
                "end_s":          round(end   / SAMPLE_RATE, 3),
                "detected":       detected,
                "detection_prob": round(det_prob, 4),
                "decoded_text":   db_match["original_text"] if db_match else decoded,
                "raw_decoded":    decoded,
                "db_match":       db_match,
            })
            start = end

        return jsonify({
            "segments":         segments,
            "total_duration_s": round(total_dur, 3),
            "window_s":         1.0,
            "sample_rate":      SAMPLE_RATE,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


@app.route("/api/audio/encode", methods=["POST"])
def api_audio_encode():
    """
    Upload an audio file + text → embed AudioSeal watermark.

    Form fields:
        file        : audio file (wav / mp3 / flac / ...)
        text        : ≤2 ASCII chars (16-bit message)
        session_id  : (optional) existing session; creates new one if absent

    Returns JSON with session_id, original_url, watermarked_url, bits_list,
    original_text, duration_s, embed_time_s.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    f    = request.files["file"]
    text = request.form.get("text", "").strip()[:2]  # max 2 ASCII chars

    if not f.filename:
        return jsonify({"error": "No selected file"}), 400

    if not _allowed_audio(f.filename):
        return jsonify({"error": f"Unsupported audio format. Supported: {', '.join(sorted(ALLOWED_AUDIO_EXT)).upper()}"}), 400

    session_id = request.form.get("session_id") or new_session_id()
    sess_dir   = os.path.join(SESSION_FOLDER, session_id)
    os.makedirs(sess_dir, exist_ok=True)

    # Save uploaded file
    orig_ext    = os.path.splitext(secure_filename(f.filename))[1].lower() or ".wav"
    upload_path = os.path.join(sess_dir, f"audio_original{orig_ext}")
    f.save(upload_path)

    wm_path = os.path.join(sess_dir, "audio_watermarked.wav")

    try:
        info = embed_audio(upload_path, text, wm_path, DEVICE)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Persist audio session meta
    import json as _json
    meta_path = os.path.join(sess_dir, "audio_meta.json")
    with open(meta_path, "w") as mf:
        _json.dump({
            "bits_list":     info["bits_list"],
            "original_text": info["original_text"],
            "orig_ext":      orig_ext,
        }, mf)

    return jsonify({
        "session_id":      session_id,
        "original_url":    f"/api/audio/session/{session_id}/original{orig_ext}",
        "watermarked_url": f"/api/audio/session/{session_id}/watermarked.wav",
        "bits_list":       info["bits_list"],
        "original_text":   info["original_text"],
        "duration_s":      info["duration_s"],
        "embed_time_s":    info["embed_time_s"],
    })


@app.route("/api/audio/session/<session_id>/<filename>")
def api_audio_file(session_id, filename):
    """Serve audio files for a session."""
    safe = secure_filename(filename)
    path = os.path.join(SESSION_FOLDER, session_id, f"audio_{safe}")
    if not os.path.isfile(path):
        # Also try without prefix
        path = os.path.join(SESSION_FOLDER, session_id, safe)
    if not os.path.isfile(path):
        abort(404)
    dl = request.args.get("dl") == "1"
    dl_name = request.args.get("filename") or safe
    return send_file(path, as_attachment=dl, download_name=dl_name if dl else None)


@app.route("/api/audio/attacks/run", methods=["POST"])
def api_audio_attacks_run():
    """
    Run audio attacks on the watermarked audio, detect watermark after each.

    JSON body: { session_id, attacks: [key, ...] }

    Returns { results: [...] } — same shape as video attack results.
    """
    import json as _json

    data       = request.get_json(force=True) or {}
    session_id = data.get("session_id", "")
    attacks    = data.get("attacks", [])

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    sess_dir  = os.path.join(SESSION_FOLDER, session_id)
    meta_path = os.path.join(sess_dir, "audio_meta.json")
    if not os.path.isfile(meta_path):
        return jsonify({"error": "Audio session not found. Re-embed first."}), 404

    with open(meta_path) as mf:
        meta = _json.load(mf)

    orig_bits  = meta["bits_list"]
    orig_text  = meta["original_text"]
    wm_path    = os.path.join(sess_dir, "audio_watermarked.wav")

    results = []
    for key in attacks:
        if key not in AUDIO_ATTACKS:
            continue
        display_name, _ = AUDIO_ATTACKS[key]
        out_path = os.path.join(sess_dir, f"audio_attacked_{key}.wav")

        try:
            proc_time = run_audio_attack(wm_path, key, out_path)
            det       = detect_audio(out_path, orig_bits, DEVICE)
            db_match  = find_nearest(det["bits_list"]) if det.get("bits_list") else None
            results.append({
                "attack_key":    key,
                "display_name":  display_name,
                "attacked_url":  f"/api/audio/session/{session_id}/attacked_{key}.wav",
                "detection_prob": det["detection_prob"],
                "bit_accuracy":  det["bit_accuracy"],
                "bits_list":     det["bits_list"],
                "decoded_text":  det["decoded_text"],
                "orig_text":     orig_text,
                "pass":          det["detection_prob"] >= 0.5,
                "process_time_s": proc_time,
                "error":         None,
                "db_match":      db_match,
            })
        except Exception as e:
            results.append({
                "attack_key":   key,
                "display_name": display_name,
                "error":        str(e),
                "pass":         False,
            })

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return jsonify({"results": results})


@app.route("/api/rs_info")
def api_rs_info():
    """Return RS parameter info for a given k."""
    try:
        k = int(request.args.get("k", DEFAULT_K))
    except ValueError:
        return jsonify({"error": "k must be integer"}), 400
    return jsonify(rs_info(k, RS_TOTAL_BYTES))


@app.route("/api/bch_info")
def api_bch_info():
    """Return fixed BCH parameter info."""
    return jsonify({**bch_info(MSG_BITS), "available": BCH_AVAILABLE})


@app.route("/api/status")
def api_status():
    """Return device info and GPU memory usage."""
    info = {"device": str(DEVICE)}
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        allocated = torch.cuda.memory_allocated(DEVICE)
        reserved  = torch.cuda.memory_reserved(DEVICE)
        total     = props.total_memory
        info.update({
            "gpu_name":        props.name,
            "gpu_total_mb":    round(total     / 1024**2),
            "gpu_reserved_mb": round(reserved  / 1024**2),
            "gpu_allocated_mb":round(allocated / 1024**2),
            "gpu_free_mb":     round((total - reserved) / 1024**2),
        })
    return jsonify(info)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(DEVICE)
        total_gb = props.total_memory / 1024**3
        print(f"GPU: {props.name}  |  VRAM: {total_gb:.1f} GB")
    else:
        print("CUDA not available — running on CPU (slower)")

    print("Pre-loading model...")
    get_model()

    if DEVICE.type == "cuda":
        alloc_mb = torch.cuda.memory_allocated(DEVICE) / 1024**2
        print(f"Model loaded. GPU memory used: {alloc_mb:.0f} MB")

    print("Starting server at http://localhost:5000")
    print("GPU status: http://localhost:5000/api/status")
    # threaded=False: PyTorch/CUDA requires single-thread per process
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)
