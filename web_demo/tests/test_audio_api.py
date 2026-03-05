"""
Test cases: Audio Watermark API

pytest web_demo/tests/test_audio_api.py -v
"""

import os
import requests
import pytest

BASE    = "http://localhost:5000"
WM_TEXT = "Hi"

ATTACKS = ["noise", "volume", "lowpass", "mp3_128k"]


# ── 1. Encode ───────────────────────────────────────────────────────────────
def test_audio_encode_session_id(audio_session):
    assert "session_id" in audio_session
    assert audio_session["session_id"]


def test_audio_encode_original_text(audio_session):
    assert audio_session.get("original_text") == WM_TEXT


def test_audio_encode_bits_list(audio_session):
    bits = audio_session.get("bits_list", [])
    assert len(bits) == 16
    assert all(b in (0, 1) for b in bits)


def test_audio_encode_urls_accessible(audio_session):
    for key in ("original_url", "watermarked_url"):
        url = BASE + audio_session[key]
        r   = requests.head(url, timeout=10)
        assert r.status_code == 200, f"{key} không truy cập được: {url}"


# ── 2. Attacks ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def audio_attack_results(audio_session):
    r = requests.post(
        f"{BASE}/api/audio/attacks/run",
        json={"session_id": audio_session["session_id"], "attacks": ATTACKS},
        timeout=120,
    )
    assert r.status_code == 200, f"audio attacks/run thất bại: {r.text[:200]}"
    return r.json().get("results", [])


def test_audio_attacks_count(audio_attack_results):
    assert len(audio_attack_results) == len(ATTACKS)


def test_audio_attacks_no_server_error(audio_attack_results):
    errors = [r for r in audio_attack_results if r.get("error")]
    assert not errors, f"Attack có lỗi: {[e['attack_key'] for e in errors]}"


def test_audio_attacks_have_detection_prob(audio_attack_results):
    for res in audio_attack_results:
        assert res.get("detection_prob") is not None, \
            f"Thiếu detection_prob: {res.get('attack_key')}"


def test_audio_attacks_volume_passes(audio_attack_results):
    """Giảm volume là attack nhẹ, kỳ vọng watermark vẫn phát hiện được."""
    res = next((r for r in audio_attack_results if r["attack_key"] == "volume"), None)
    assert res is not None
    assert res.get("pass"), \
        f"volume không pass: det={res.get('detection_prob')}"


# ── 3. Temporal detection ────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def audio_temporal_result(audio_path):
    with open(audio_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/audio/detect_temporal",
            files={"audio": (os.path.basename(audio_path), f, "audio/wav")},
            timeout=120,
        )
    assert r.status_code == 200, f"audio/detect_temporal thất bại: {r.text[:300]}"
    return r.json()


def test_audio_temporal_has_segments(audio_temporal_result):
    assert len(audio_temporal_result.get("segments", [])) > 0


def test_audio_temporal_segment_fields(audio_temporal_result):
    for seg in audio_temporal_result["segments"]:
        assert "start_s"        in seg
        assert "end_s"          in seg
        assert "detected"       in seg
        assert "detection_prob" in seg


def test_audio_temporal_duration(audio_temporal_result):
    assert audio_temporal_result.get("total_duration_s", 0) > 0
