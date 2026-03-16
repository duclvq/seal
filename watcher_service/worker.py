"""
Video + audio watermark embedding worker.
Uses: custom VideoSeal model, BCH-256 ECC, center_mask=True, AudioSeal for audio.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import av
import torch

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
_oom_splits = 0  # counter for OOM auto-splits (per-video, reset in embed_to_file)
CRF        = "18"   # near-visually-lossless H.264


# ─────────────────────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_video_model(ckpt_path: str, device: torch.device):
    """Load custom VideoSeal checkpoint."""
    model = setup_model_from_checkpoint(ckpt_path)
    model.eval().to(device)
    return model


def load_audio_model(device: torch.device):
    """Load AudioSeal generator (16-bit). Returns None if unavailable."""
    try:
        from audioseal import AudioSeal
        gen = AudioSeal.load_generator("audioseal_wm_16bits").eval().to(device)
        return gen
    except Exception:
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


def _embed_batch(frames: torch.Tensor, model, msg_gpu, device) -> torch.Tensor:
    """Embed a batch of frames. Auto-splits on OOM (recursive halving)."""
    import logging
    _log = logging.getLogger(__name__)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    video_gpu = None
    outputs = None
    try:
        video_gpu = frames.to(device)
        with torch.no_grad():
            outputs = model.embed(video_gpu, msgs=msg_gpu, is_video=True)
        result = outputs["imgs_w"].cpu()
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
        part1 = _embed_batch(frames[:half], model, msg_gpu, device)
        part2 = _embed_batch(frames[half:], model, msg_gpu, device)
        return torch.cat([part1, part2], dim=0)


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
) -> tuple:
    video_chunk = torch.stack(chunk).float() / 255.0
    chunk.clear()  # free raw frame memory immediately

    wm_chunk = _embed_batch(video_chunk, model, msg_gpu, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    del video_chunk
    vid_np   = (wm_chunk.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
    del wm_chunk

    if out_stream is None:
        H = vid_np.shape[1] - vid_np.shape[1] % 2
        W = vid_np.shape[2] - vid_np.shape[2] % 2
        out_stream         = out_container.add_stream("h264", rate=int(fps))
        out_stream.width   = W
        out_stream.height  = H
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": CRF, "preset": "fast"}
        h_out, w_out = H, W

    for frame_np in vid_np[:, :h_out, :w_out, :]:
        av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
        for packet in out_stream.encode(av_frame):
            out_container.mux(packet)

    return out_stream, h_out, w_out


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
) -> dict:
    """
    Full pipeline: embed VideoSeal + (optionally) AudioSeal watermark.

    Args:
        input_path:     source video file
        output_path:    destination (same filename/ext expected by caller)
        video_model:    loaded VideoSeal custom model
        watermark_text: text to embed (BCH-256 for video, first 2 chars for audio)
        device:         torch.device
        audio_model:    AudioSeal generator, or None to skip audio watermark

    Returns:
        dict with keys: fps, total_frames, has_audio, resolution, oom_splits
    """
    global _oom_splits
    _oom_splits = 0  # reset per video

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

    # ── Extract original audio ────────────────────────────────────────────────
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

    import logging, time as _time
    _log = logging.getLogger(__name__)

    fps          = 24.0
    total_frames = 0
    h_out = w_out = None
    out_stream    = None
    msg_gpu       = msg_tensor.to(device)
    _filename     = Path(input_path).name

    try:
        with av.open(tmp_vid, mode="w") as out_container:
            with av.open(av_path) as in_container:
                in_stream = in_container.streams.video[0]
                rate = in_stream.average_rate or in_stream.guessed_rate
                if rate:
                    fps = float(rate)
                n_total = in_stream.frames or 0
                res = f"{in_stream.width}x{in_stream.height}"
                _log.info(f"[EMBED] {_filename}: {res}, ~{n_total}f, fps={fps:.1f}, chunk={CHUNK_SIZE}")
                in_stream.thread_type = "AUTO"

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
                        )
                        total_frames += CHUNK_SIZE
                        dt = _time.time() - t0
                        gpu_s = ""
                        if device.type == "cuda":
                            alloc = torch.cuda.memory_allocated(device) / 1024**2
                            peak  = torch.cuda.max_memory_allocated(device) / 1024**2
                            util  = f"alloc={alloc:.0f}MB peak={peak:.0f}MB"
                            gpu_s = f"  [{util}]"
                        _log.info(f"  [{_filename}] {total_frames}/{n_total}f  ({dt:.1f}s, {CHUNK_SIZE/dt:.1f}f/s){gpu_s}")
                        chunk = []

                if chunk:
                    out_stream, h_out, w_out = _flush_chunk(
                        chunk, video_model, msg_gpu, device,
                        out_container, out_stream, fps, h_out, w_out,
                    )
                    total_frames += len(chunk)

            if out_stream:
                for packet in out_stream.encode():
                    out_container.mux(packet)
    finally:
        del msg_gpu
        if tmp_converted and os.path.exists(tmp_converted):
            os.unlink(tmp_converted)

    # ── Mux audio + finalize container ───────────────────────────────────────
    import subprocess
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

    return {
        "fps": fps,
        "total_frames": total_frames,
        "has_audio": has_audio,
        "resolution": res,
        "oom_splits": _oom_splits,
    }
