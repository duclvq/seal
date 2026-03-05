"""
Shared pytest fixtures cho tất cả test API.
"""

import os
import math
import struct
import pytest
import requests

BASE       = "http://localhost:5000"
VIDEO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../input_video/[142MB_19s]_DJI_20260104091736_0149_D_compressed.mp4")
)
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "test_audio.wav")
WM_VIDEO   = "Test"
WM_AUDIO   = "Hi"


# ── Server check ────────────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def check_server():
    try:
        requests.get(BASE, timeout=5)
    except Exception:
        pytest.exit(f"Server không chạy tại {BASE}  →  cd web_demo && python app.py", returncode=1)


# ── Video fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def video_path():
    if not os.path.exists(VIDEO_PATH):
        pytest.skip(f"Không tìm thấy video: {VIDEO_PATH}")
    return VIDEO_PATH


@pytest.fixture(scope="session")
def video_session(video_path):
    """Embed watermark một lần, dùng chung cho tất cả test video."""
    with open(video_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/encode",
            files={"video": (os.path.basename(video_path), f, "video/mp4")},
            data={"text": WM_VIDEO, "ecc_type": "bch"},
            timeout=300,
        )
    assert r.status_code == 200, f"Encode thất bại: {r.text[:300]}"
    return r.json()


# ── Audio fixtures ──────────────────────────────────────────────────────────
def _make_wav(path, duration_s=5, sample_rate=16000, freq=440):
    """Tạo file WAV sine wave đơn giản (không cần thư viện ngoài)."""
    n      = duration_s * sample_rate
    amp    = int(32767 * 0.5)
    frames = [int(amp * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)]
    with open(path, "wb") as f:
        data_size = n * 2
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate*2, 2, 16))
        f.write(b"data"); f.write(struct.pack("<I", data_size))
        for s in frames:
            f.write(struct.pack("<h", s))


@pytest.fixture(scope="session")
def audio_path():
    if not os.path.exists(AUDIO_PATH):
        _make_wav(AUDIO_PATH)
    return AUDIO_PATH


@pytest.fixture(scope="session")
def audio_session(audio_path):
    """Embed audio watermark một lần, dùng chung cho tất cả test audio."""
    with open(audio_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/audio/encode",
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
            data={"text": WM_AUDIO},
            timeout=120,
        )
    assert r.status_code == 200, f"Audio encode thất bại: {r.text[:300]}"
    return r.json()
