# API Test Suite

Test tự động cho các endpoint của VideoSeal Web Demo.

## Yêu cầu

- Server đang chạy tại `http://localhost:5000`
- Python 3.8+
- `pip install pytest requests`

## Cấu trúc

```
tests/
├── conftest.py        # Fixtures dùng chung (session_id, video/audio path, WAV generator)
├── pytest.ini         # Cấu hình pytest
├── test_video_api.py  # Test các endpoint video
├── test_audio_api.py  # Test các endpoint audio
└── run_all.py         # Chạy không cần pytest
```

## Chạy test

```bash
# Khởi động server (terminal riêng)
cd web_demo
python app.py

# Chạy tất cả
cd web_demo/tests
pytest

# Chỉ video
pytest test_video_api.py -v

# Chỉ audio
pytest test_audio_api.py -v

# Một test cụ thể
pytest test_video_api.py::test_attacks_brightness_passes -v

# Hiện output print (hữu ích khi debug)
pytest -s
```

## Chạy test format

```bash
pytest test_formats.py -v
```

Tự động tạo file test (3s) trong `tests/tmp_formats/` bằng ffmpeg rồi upload từng format. Cần ffmpeg trên PATH, nếu không có thì toàn bộ file test bị skip.

| Format video | Ghi chú |
|---|---|
| mp4, avi, mov, mkv, webm | PyAV đọc trực tiếp |
| mxf, vob, wmv, f4v, m2ts | Tự động convert sang mp4 qua ffmpeg trước |

| Format audio | |
|---|---|
| wav, mp3, flac, ogg, aac, m4a, wma, opus | torchaudio đọc trực tiếp |

## Endpoints được test

### Video (`test_video_api.py`)

| Test | Endpoint | Mô tả |
|------|----------|-------|
| `test_status` | `GET /api/status` | Server hoạt động bình thường |
| `test_encode_*` | `POST /api/encode` | Nhúng watermark BCH, kiểm tra session, bits, URL |
| `test_attacks_*` | `POST /api/attacks/run` | 4 attack: brightness, h264, gaussian_noise, crop |
| `test_temporal_upload_*` | `POST /api/video/detect_temporal` | Phát hiện theo thời gian (upload file) |
| `test_temporal_session_*` | `POST /api/video/detect_temporal` | Phát hiện theo thời gian (session mode) |

### Audio (`test_audio_api.py`)

| Test | Endpoint | Mô tả |
|------|----------|-------|
| `test_audio_encode_*` | `POST /api/audio/encode` | Nhúng watermark 16-bit, kiểm tra session, bits, URL |
| `test_audio_attacks_*` | `POST /api/audio/attacks/run` | 4 attack: noise, volume, lowpass, mp3_128k |
| `test_audio_temporal_*` | `POST /api/audio/detect_temporal` | Phát hiện theo từng giây |

## Dữ liệu đầu vào

**Video:** `input_video/[142MB_19s]_DJI_20260104091736_0149_D_compressed.mp4`
- Test tự động tìm file này từ thư mục gốc project
- Nếu không tìm thấy, test sẽ bị skip

**Audio:** `tests/test_audio.wav`
- Nếu không có sẵn, tự động sinh file WAV sine wave 5 giây (440 Hz, 16kHz mono)
- Không cần cài thêm thư viện

## Chiến lược fixture

```
scope="session"   video_session, audio_session   # encode 1 lần duy nhất
scope="module"    attack_results, temporal_*     # gọi API 1 lần, nhiều test dùng chung
scope="function"  (mặc định)                    # mỗi test độc lập
```

Nhờ đó, mỗi lần chạy toàn bộ suite chỉ thực hiện **1 lần encode video** và **1 lần encode audio**, tiết kiệm thời gian đáng kể.

## Ví dụ output

```
tests/test_video_api.py::test_status                          PASSED
tests/test_video_api.py::test_encode_returns_session          PASSED
tests/test_video_api.py::test_encode_original_text            PASSED
tests/test_video_api.py::test_encode_bits_list                PASSED
tests/test_video_api.py::test_encode_has_video_urls           PASSED
tests/test_video_api.py::test_encode_video_urls_accessible    PASSED
tests/test_video_api.py::test_attacks_count                   PASSED
tests/test_video_api.py::test_attacks_no_server_error         PASSED
tests/test_video_api.py::test_attacks_have_bit_accuracy       PASSED
tests/test_video_api.py::test_attacks_brightness_passes       PASSED
tests/test_video_api.py::test_temporal_upload_has_segments    PASSED
...
```
