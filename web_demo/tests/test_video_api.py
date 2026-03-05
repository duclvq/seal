"""
Test cases: Video Watermark API

pytest web_demo/tests/test_video_api.py -v
"""

import os
import requests
import pytest

BASE     = "http://localhost:5000"
WM_TEXT  = "Test"


# ── 1. Server status ────────────────────────────────────────────────────────
def test_status():
    r = requests.get(f"{BASE}/api/status", timeout=10)
    assert r.status_code == 200
    body = r.json()
    assert "model" in body or "status" in body or body  # bất kỳ JSON hợp lệ


# ── 2. Encode ───────────────────────────────────────────────────────────────
def test_encode_returns_session(video_session):
    assert "session_id" in video_session
    assert video_session["session_id"]


def test_encode_original_text(video_session):
    assert video_session.get("text") == WM_TEXT


def test_encode_bits_list(video_session):
    bits = video_session.get("bits_list", [])
    assert len(bits) > 0
    assert all(b in (0, 1) for b in bits)


def test_encode_has_video_urls(video_session):
    assert "original_url" in video_session
    assert "watermarked_url" in video_session


def test_encode_video_urls_accessible(video_session):
    for key in ("original_url", "watermarked_url"):
        url = BASE + video_session[key]
        r   = requests.head(url, timeout=10)
        assert r.status_code == 200, f"{key} không truy cập được: {url}"


# ── 3. Attacks ──────────────────────────────────────────────────────────────
ATTACKS = ["brightness", "h264", "gaussian_noise", "crop"]


@pytest.fixture(scope="module")
def attack_results(video_session):
    r = requests.post(
        f"{BASE}/api/attacks/run",
        json={"session_id": video_session["session_id"], "attacks": ATTACKS},
        timeout=300,
    )
    assert r.status_code == 200, f"attacks/run thất bại: {r.text[:200]}"
    return r.json().get("results", [])


def test_attacks_count(attack_results):
    assert len(attack_results) == len(ATTACKS)


def test_attacks_no_server_error(attack_results):
    errors = [r for r in attack_results if r.get("error")]
    assert not errors, f"Attack có lỗi server: {[e['attack_key'] for e in errors]}"


def test_attacks_have_bit_accuracy(attack_results):
    for res in attack_results:
        assert res.get("bit_accuracy") is not None, \
            f"Thiếu bit_accuracy trong kết quả: {res.get('attack_key')}"


def test_attacks_brightness_passes(attack_results):
    """brightness là attack nhẹ, kỳ vọng watermark vẫn phát hiện được."""
    res = next((r for r in attack_results if r["attack"] == "brightness"), None)
    assert res is not None
    assert res.get("pass"), \
        f"brightness không pass: bit_acc={res.get('bit_accuracy')}"


# ── 4. Temporal detection — upload mode ─────────────────────────────────────
@pytest.fixture(scope="module")
def temporal_upload_result(video_path):
    with open(video_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/video/detect_temporal",
            files={"video": (os.path.basename(video_path), f, "video/mp4")},
            timeout=300,
        )
    assert r.status_code == 200, f"detect_temporal (upload) thất bại: {r.text[:300]}"
    return r.json()


def test_temporal_upload_has_segments(temporal_upload_result):
    segments = temporal_upload_result.get("segments", [])
    assert len(segments) > 0


def test_temporal_upload_segment_fields(temporal_upload_result):
    for seg in temporal_upload_result["segments"]:
        assert "start_s"    in seg
        assert "end_s"      in seg
        assert "detected"   in seg
        assert "bit_accuracy" in seg


def test_temporal_upload_duration(temporal_upload_result):
    assert temporal_upload_result.get("total_duration_s", 0) > 0


# ── 5. Temporal detection — session mode ────────────────────────────────────
@pytest.fixture(scope="module")
def temporal_session_result(video_session):
    r = requests.post(
        f"{BASE}/api/video/detect_temporal",
        json={"session_id": video_session["session_id"]},
        timeout=300,
    )
    assert r.status_code == 200, f"detect_temporal (session) thất bại: {r.text[:300]}"
    return r.json()


def test_temporal_session_has_segments(temporal_session_result):
    assert len(temporal_session_result.get("segments", [])) > 0


def test_temporal_session_majority_detected(temporal_session_result):
    """Video vừa embed → đa số segment phải có watermark."""
    segs     = temporal_session_result["segments"]
    detected = sum(1 for s in segs if s.get("detected"))
    ratio    = detected / len(segs)
    assert ratio >= 0.5, \
        f"Chỉ {detected}/{len(segs)} segment có watermark ({ratio:.0%})"
