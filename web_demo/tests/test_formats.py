"""
Test cases: định dạng video và audio được hỗ trợ.

Tự động chuyển đổi video gốc sang các format khác bằng ffmpeg,
rồi gửi từng format lên API để kiểm tra.

pytest web_demo/tests/test_formats.py -v
"""

import math
import os
import struct
import subprocess
import pytest
import requests

BASE       = "http://localhost:5000"
SOURCE_MP4 = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../input_video/[142MB_19s]_DJI_20260104091736_0149_D_compressed.mp4")
)
TMP_DIR    = os.path.join(os.path.dirname(__file__), "tmp_formats")

# ── Format definitions ───────────────────────────────────────────────────────

VIDEO_FORMATS = [
    # (ext, ffmpeg_extra_args, mime)
    ("mp4",  [],                                            "video/mp4"),
    ("avi",  ["-c:v", "libx264"],                          "video/avi"),
    ("mov",  ["-c:v", "libx264"],                          "video/quicktime"),
    ("mkv",  ["-c:v", "libx264"],                          "video/x-matroska"),
    ("webm", ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "33"], "video/webm"),
    ("mxf",  ["-c:v", "mpeg2video", "-f", "mxf"],         "application/mxf"),
]

AUDIO_FORMATS = [
    # (ext, ffmpeg_extra_args, mime)
    ("wav",  [],                                    "audio/wav"),
    ("mp3",  ["-b:a", "128k"],                      "audio/mpeg"),
    ("flac", [],                                    "audio/flac"),
    ("ogg",  ["-c:a", "libvorbis"],                 "audio/ogg"),
    ("aac",  ["-b:a", "128k"],                      "audio/aac"),
    ("m4a",  ["-c:a", "aac", "-b:a", "128k"],       "audio/mp4"),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _make_short_mp4(src: str, dst: str, duration_s: int = 3):
    """Cắt 3 giây đầu của video gốc thành mp4 nhỏ để test nhanh."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-t", str(duration_s),
         "-c:v", "libx264", "-an", dst],
        capture_output=True, check=True,
    )


def _convert(src: str, dst: str, extra_args: list):
    """Chuyển đổi file sang format khác bằng ffmpeg."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", src, "-t", "3"] + extra_args + ["-an", dst]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({dst}):\n{result.stderr.decode(errors='replace')[-500:]}"
        )


def _make_wav(path: str, duration_s: int = 5, sample_rate: int = 16000, freq: int = 440):
    """Tạo file WAV sine wave (không cần thư viện ngoài)."""
    n      = duration_s * sample_rate
    amp    = int(32767 * 0.5)
    frames = [int(amp * math.sin(2 * math.pi * freq * i / sample_rate)) for i in range(n)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        data_size = n * 2
        f.write(b"RIFF"); f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt "); f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate*2, 2, 16))
        f.write(b"data"); f.write(struct.pack("<I", data_size))
        for s in frames:
            f.write(struct.pack("<h", s))


def _convert_audio(src: str, dst: str, extra_args: list):
    """Chuyển đổi WAV → audio format khác bằng ffmpeg."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", src] + extra_args + [dst]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({dst}):\n{result.stderr.decode(errors='replace')[-500:]}"
        )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def require_ffmpeg():
    if not _ffmpeg_available():
        pytest.skip("ffmpeg không có trên hệ thống — bỏ qua test format")


@pytest.fixture(scope="session")
def source_video():
    if not os.path.exists(SOURCE_MP4):
        pytest.skip(f"Không tìm thấy video gốc: {SOURCE_MP4}")
    short = os.path.join(TMP_DIR, "source_3s.mp4")
    if not os.path.exists(short):
        _make_short_mp4(SOURCE_MP4, short, duration_s=3)
    return short


@pytest.fixture(scope="session")
def converted_videos(source_video):
    """Dict ext → path cho tất cả video format."""
    paths = {}
    for ext, extra, _ in VIDEO_FORMATS:
        dst = os.path.join(TMP_DIR, f"test_video.{ext}")
        if not os.path.exists(dst):
            try:
                if ext == "mp4":
                    import shutil; shutil.copy(source_video, dst)
                else:
                    _convert(source_video, dst, extra)
            except RuntimeError as e:
                paths[ext] = None
                print(f"\n  [WARN] Không tạo được {ext}: {e}")
                continue
        paths[ext] = dst
    return paths


@pytest.fixture(scope="session")
def converted_audios():
    """Dict ext → path cho tất cả audio format (source: generated WAV)."""
    base_wav = os.path.join(TMP_DIR, "test_audio.wav")
    if not os.path.exists(base_wav):
        _make_wav(base_wav)

    paths = {}
    for ext, extra, _ in AUDIO_FORMATS:
        dst = os.path.join(TMP_DIR, f"test_audio.{ext}")
        if not os.path.exists(dst):
            try:
                if ext == "wav":
                    import shutil; shutil.copy(base_wav, dst)
                else:
                    _convert_audio(base_wav, dst, extra)
            except RuntimeError as e:
                paths[ext] = None
                print(f"\n  [WARN] Không tạo được {ext}: {e}")
                continue
        paths[ext] = dst
    return paths


# ── Video format tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ext,_extra,mime", VIDEO_FORMATS, ids=[f[0] for f in VIDEO_FORMATS])
def test_video_format_encode(ext, _extra, mime, converted_videos):
    path = converted_videos.get(ext)
    if path is None:
        pytest.skip(f"Không tạo được file {ext}")

    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/encode",
            files={"video": (f"test.{ext}", f, mime)},
            data={"text": "Hi", "ecc_type": "bch"},
            timeout=120,
        )

    assert r.status_code == 200, \
        f"[{ext}] HTTP {r.status_code}: {r.text[:300]}"

    data = r.json()
    assert data.get("session_id"), f"[{ext}] Thiếu session_id"
    assert data.get("bits_list"),  f"[{ext}] Thiếu bits_list"


@pytest.mark.parametrize("ext,_extra,mime", VIDEO_FORMATS, ids=[f[0] for f in VIDEO_FORMATS])
def test_video_format_temporal(ext, _extra, mime, converted_videos):
    path = converted_videos.get(ext)
    if path is None:
        pytest.skip(f"Không tạo được file {ext}")

    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/video/detect_temporal",
            files={"video": (f"test.{ext}", f, mime)},
            timeout=120,
        )

    assert r.status_code == 200, \
        f"[{ext}] temporal HTTP {r.status_code}: {r.text[:300]}"

    data = r.json()
    assert len(data.get("segments", [])) > 0, f"[{ext}] Không có segment"


# ── Audio format tests ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ext,_extra,mime", AUDIO_FORMATS, ids=[f[0] for f in AUDIO_FORMATS])
def test_audio_format_encode(ext, _extra, mime, converted_audios):
    path = converted_audios.get(ext)
    if path is None:
        pytest.skip(f"Không tạo được file {ext}")

    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/audio/encode",
            files={"file": (f"test.{ext}", f, mime)},
            data={"text": "Hi"},
            timeout=60,
        )

    assert r.status_code == 200, \
        f"[{ext}] HTTP {r.status_code}: {r.text[:300]}"

    data = r.json()
    assert data.get("session_id"),        f"[{ext}] Thiếu session_id"
    assert len(data.get("bits_list", [])) == 16, f"[{ext}] bits_list sai độ dài"


@pytest.mark.parametrize("ext,_extra,mime", AUDIO_FORMATS, ids=[f[0] for f in AUDIO_FORMATS])
def test_audio_format_temporal(ext, _extra, mime, converted_audios):
    path = converted_audios.get(ext)
    if path is None:
        pytest.skip(f"Không tạo được file {ext}")

    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/audio/detect_temporal",
            files={"audio": (f"test.{ext}", f, mime)},
            timeout=60,
        )

    assert r.status_code == 200, \
        f"[{ext}] temporal HTTP {r.status_code}: {r.text[:300]}"

    data = r.json()
    assert len(data.get("segments", [])) > 0, f"[{ext}] Không có segment"


# ── MXF attack tests ─────────────────────────────────────────────────────────

MXF_ATTACKS = ["brightness", "h264", "gaussian_noise", "crop"]
_MXF_MIME   = "application/mxf"
_MXF_EXT    = "mxf"
_MXF_EXTRA  = ["-c:v", "mpeg2video", "-f", "mxf"]


@pytest.fixture(scope="module")
def mxf_session(source_video):
    """Encode MXF watermark một lần, dùng chung cho tất cả MXF attack tests."""
    mxf_path = os.path.join(TMP_DIR, f"test_video.{_MXF_EXT}")
    if not os.path.exists(mxf_path):
        _convert(source_video, mxf_path, _MXF_EXTRA)

    with open(mxf_path, "rb") as f:
        r = requests.post(
            f"{BASE}/api/encode",
            files={"video": (f"test.{_MXF_EXT}", f, _MXF_MIME)},
            data={"text": "MX", "ecc_type": "bch"},
            timeout=120,
        )
    assert r.status_code == 200, f"MXF encode thất bại: {r.text[:300]}"
    return r.json()


@pytest.fixture(scope="module")
def mxf_attack_results(mxf_session):
    """Chạy 4 attacks trên video MXF đã nhúng watermark."""
    r = requests.post(
        f"{BASE}/api/attacks/run",
        json={"session_id": mxf_session["session_id"], "attacks": MXF_ATTACKS},
        timeout=300,
    )
    assert r.status_code == 200, f"MXF attacks/run thất bại: {r.text[:200]}"
    return r.json().get("results", [])


def test_mxf_attack_count(mxf_attack_results):
    assert len(mxf_attack_results) == len(MXF_ATTACKS)


def test_mxf_attack_no_server_error(mxf_attack_results):
    errors = [r for r in mxf_attack_results if r.get("error")]
    assert not errors, f"MXF attack có lỗi: {[e.get('attack') for e in errors]}"


def test_mxf_attack_have_bit_accuracy(mxf_attack_results):
    for res in mxf_attack_results:
        assert res.get("bit_accuracy") is not None, \
            f"Thiếu bit_accuracy: {res.get('attack')}"


def test_mxf_attack_brightness_passes(mxf_attack_results):
    """Brightness là attack nhẹ — watermark phải còn nguyên."""
    res = next((r for r in mxf_attack_results if r["attack"] == "brightness"), None)
    assert res is not None, "Không tìm thấy kết quả brightness"
    assert res.get("pass"), \
        f"MXF brightness không pass: bit_acc={res.get('bit_accuracy'):.3f}"


@pytest.mark.parametrize("attack_name", MXF_ATTACKS)
def test_mxf_attack_returns_result(attack_name, mxf_attack_results):
    """Mỗi attack phải trả về kết quả hợp lệ."""
    res = next((r for r in mxf_attack_results if r["attack"] == attack_name), None)
    assert res is not None, f"Không tìm thấy kết quả cho attack '{attack_name}'"
    assert 0.0 <= res.get("bit_accuracy", -1) <= 1.0, \
        f"bit_accuracy ngoài khoảng [0,1]: {res.get('bit_accuracy')}"


# ── Unsupported format test ───────────────────────────────────────────────────

def test_unsupported_video_format():
    r = requests.post(
        f"{BASE}/api/encode",
        files={"video": ("test.xyz", b"fake data", "application/octet-stream")},
        data={"text": "Hi", "ecc_type": "bch"},
        timeout=10,
    )
    assert r.status_code == 400
    assert "error" in r.json()


def test_unsupported_audio_format():
    r = requests.post(
        f"{BASE}/api/audio/encode",
        files={"file": ("test.xyz", b"fake data", "application/octet-stream")},
        data={"text": "Hi"},
        timeout=10,
    )
    assert r.status_code == 400
    assert "error" in r.json()
