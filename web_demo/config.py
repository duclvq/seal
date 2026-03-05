import os
import torch

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
SESSION_FOLDER = os.path.join(BASE_DIR, "sessions")
ALLOWED_VIDEO_EXT = {
    "mp4", "avi", "mov", "mkv",          # common
    "mxf", "webm", "flv", "ts", "m2ts",  # broadcast / streaming
    "m4v", "wmv", "3gp", "f4v", "vob",   # misc
}
ALLOWED_AUDIO_EXT = {
    "wav", "mp3", "flac", "ogg",
    "aac", "m4a", "wma", "opus",
}
ALLOWED_EXT = ALLOWED_VIDEO_EXT  # backward compat
MAX_CONTENT_MB = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model selection ────────────────────────────────────────────────────────
# "videoseal"  — 256 bits  (32 bytes)
# "pixelseal"  — 256 bits  (32 bytes)
# "chunkyseal" — 1024 bits (128 bytes)
MODEL_CARD = "videoseal"

# Derived from model capacity — auto-adjusts RS params
MSG_BITS       = 1024 if MODEL_CARD == "chunkyseal" else 256
RS_TOTAL_BYTES = MSG_BITS // 8        # 128 for chunkyseal, 32 for others
DEFAULT_K      = 100 if MODEL_CARD == "chunkyseal" else 20

MAX_FRAMES = 90   # cap ~3s @ 30fps for demo
# BCH is fixed: 4 data bytes + 28 ECC bytes = 32 bytes, corrects ≤28 bit errors
# (BCH always uses first 256 bits; for chunkyseal the remaining bits are zero-padded)
