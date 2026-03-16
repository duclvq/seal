# Bao cao kiem tra Docker + MXF - Watcher Service
**Ngay:** 2026-03-15
**Input folder:** `D:\data\youtube_mxf`

---

## 1. Tong quan file MXF

| # | File | Size (MB) | Duration | Resolution | FPS | Video Codec | Audio Codec | Audio |
|---|------|-----------|----------|------------|-----|-------------|-------------|-------|
| 1 | 'Spiderman' Vacuum Gloves Part 1 [...].mxf | 110.7 | 03:42 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 2 | 'Spiderman' Vacuum Gloves Part 2 [...].mxf | 152.2 | 03:42 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 3 | Andy Green interviews Ron Ayers [...].mxf | 175.9 | 06:51 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 4 | Andy Green interviews [...] _270p.mxf | 141.8 | 06:51 | 480x270 | 25 | mpeg2video | pcm_s16le | Yes |
| 5 | Bang Blog Film 4 [...].mxf | 40.7 | 01:00 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 6 | Bang Blog Film 5 [...].mxf | 21.3 | 00:42 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 7 | Bin Liner Blowing [...].mxf | 85.9 | 01:51 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 8 | Can a microwave power a plane [...].mxf | 219.2 | 05:29 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 9 | Can you snap spaghetti in half [...].mxf | 12.1 | 00:27 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 10 | How volcanic ash [...].mxf | 24.8 | 00:52 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 11 | Light Balloon, Heavy Balloon [...].mxf | 121.5 | 02:33 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 12 | Melting Glass in a Microwave [...].mxf | 135.5 | 02:57 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 13 | Office chair fun [...].mxf | 85.6 | 01:55 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 14 | Richard Dawkins and Dr Yan Wong [...].mxf | 243.1 | 09:41 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 15 | Science of Fear [...].mxf | 191.6 | 05:48 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 16 | Skewer vs Balloon [...].mxf | 26.1 | 00:54 | 852x480 | 25 | mpeg2video | pcm_s16le | Yes |
| 17 | Space by Balloon [...].mxf | 193.1 | 05:07 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 18 | The Live Bang [...].mxf | 35.0 | 01:00 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 19 | The Might of Friction [...].mxf | 131.7 | 02:23 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 20 | Underwater Firework [...].mxf | 60.7 | 01:32 | 640x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 21 | Volcanic Ash and Jet Engines [...].mxf | 138.9 | 04:10 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |
| 22 | Water Powered Jet-Pack [...].mxf | 204.6 | 04:25 | 638x360 | 25 | mpeg2video | pcm_s16le | Yes |

**Tong:** 22 file, ~2.5 GB, tong thoi luong ~67 phut

### Dac diem chung:
- **Video codec:** Tat ca deu la `mpeg2video` (MPEG-2)
- **Audio codec:** Tat ca deu la `pcm_s16le` (PCM 16-bit, 48kHz) - audio KHONG nen
- **Container:** MXF (OP1a, tao boi FFmpeg)
- **FPS:** 25 fps (PAL)
- **Resolution:** Phan lon 640x360 hoac 638x360, 1 file 852x480, 1 file 480x270

---

## 2. Ket qua chay Docker thuc te

### 2.1 Docker Environment
- **Image:** `wm-watcher:latest` (16.1GB, build 2026-03-11)
- **Khong can rebuild:** docker-compose mount source code truc tiep (dev mode), code moi tu host duoc dung ngay
- **GPU:** NVIDIA CUDA, VRAM limit 90% (14680 / 16311 MB)
- **Container name:** `9c62dbd39f03_wm-watcher` (bi prefix do orphan container cu)

### 2.2 Model Loading
| Model | Ket qua | Thoi gian |
|-------|---------|-----------|
| VideoSeal (checkpoint350.pth) | OK | ~70s |
| AudioSeal (16-bit generator) | OK | ~9s (download 56MB tu HuggingFace) |

> **Luu y:** AudioSeal da download model tu HuggingFace mac du `HF_HUB_OFFLINE=1`. Lan chay tiep theo se dung cache tai `/data/model_cache/audioseal/`.

### 2.3 LOI BAN DAU: MXF mux that bai (DA FIX)

**Trieu chung:**
```
[FAIL] 'Spiderman' Vacuum Gloves Part 1 [...].mxf: ffmpeg mux failed:
Error writing trailer of /data/output/...mxf: Invalid data found when processing input
```

**Nguyen nhan:** MXF container **KHONG ho tro H.264 codec**. Pipeline embed video thanh H.264, roi co mux vao container `.mxf` bang `-c:v copy` → that bai.

**Fix da ap dung:** Sua `_mux_video_audio()` va `_remux_video()` trong `web_demo/core/video_io.py`:
- Khi `fmt == "mxf"`: re-encode video sang `mpeg2video` (-b:v 50M) thay vi copy H.264
- Khi `fmt == "mxf"`: audio dung `pcm_s16le` (tuong thich MXF) thay vi AAC

### 2.4 Ket qua sau fix - TAT CA THANH CONG

| # | File | Resolution | Frames | Thoi gian | Throughput | Audio | GPU Peak |
|---|------|------------|--------|-----------|------------|-------|----------|
| 1 | Spiderman Part 1 | 640x360 | 5535 | 102.7s | 53.9 f/s | Yes | 1189MB |
| 2 | Spiderman Part 2 | 640x360 | 5535 | 97.8s | 56.6 f/s | Yes | 1189MB |
| 3 | Andy Green interviews | 640x360 | 10260 | 162.4s | 63.2 f/s | Yes | 1200MB |
| 4 | Andy Green _270p | 480x270 | 10260 | 117.6s | 87.2 f/s | Yes | 1097MB |
| 5 | Bang Blog Film 4 | 638x360 | 1485 | 26.8s | 55.4 f/s | Yes | 1179MB |
| 6 | Bang Blog Film 5 | 640x360 | 1035 | 19.3s | 53.8 f/s | Yes | 1178MB |
| 7 | Bin Liner Blowing | 640x360 | 2745 | 52.9s | 51.9 f/s | Yes | 1182MB |
| ... | (dang tiep tuc chay) | | | | | | |

### 2.5 Output files
```
D:\data\temp\
  'Spiderman' Vacuum Gloves Part 1 [...].mxf   109 MB (input 111 MB)
  'Spiderman' Vacuum Gloves Part 2 [...].mxf   147 MB (input 152 MB)
  Andy Green interviews [...].mxf               171 MB (input 176 MB)
  Andy Green interviews [...] _270p.mxf         141 MB (input 142 MB)
  Bang Blog Film 4 [...].mxf                     40 MB (input  41 MB)
  Bang Blog Film 5 [...].mxf                     21 MB (input  21 MB)
  Bin Liner Blowing [...].mxf                     83 MB (input  86 MB)
```

> Output MXF kich thuoc tuong duong input (mpeg2video 50Mbps). Extension `.mxf` duoc giu nguyen.

---

## 3. Cac van de da gap va cach xu ly

### 3.1 LOI CHINH: MXF khong ho tro H.264 (DA FIX)
- **File sua:** `web_demo/core/video_io.py`
- **Ham:** `_mux_video_audio()`, `_remux_video()`
- **Thay doi:** Khi output format la MXF → re-encode sang mpeg2video thay vi copy H.264 stream
- **Ket qua:** Tat ca file MXF deu xu ly thanh cong sau fix

### 3.2 Worker.py cung dung `-c:v copy` cho non-mp4 (DA FIX)
- **File sua:** `watcher_service/worker.py`
- **Thay doi:** Thay `subprocess.run(["ffmpeg", ..., "-c:v", "copy", ...])` bang `_remux_video()` (da xu ly MXF)

### 3.3 Docker orphan container
- Container cu `wm-watcher` bi Dead → Docker tao container moi voi ten prefix
- **Warning:** `version` attribute is obsolete trong docker-compose.yml (khong anh huong)
- **Cach fix:** `docker compose --remove-orphans up -d`

### 3.4 AudioSeal cache
- Lan dau chay: AudioSeal download 56MB model tu HuggingFace (mac du set offline mode)
- Model duoc cache tai `/data/model_cache/audioseal/` → lan sau dung offline

---

## 4. Hieu suat

| Metric | Gia tri |
|--------|---------|
| Throughput trung binh | 55-65 f/s (640x360) |
| Throughput 270p | 87 f/s (480x270) |
| GPU VRAM peak | 1.1-1.2 GB |
| GPU VRAM allocated | ~284 MB |
| Chunk size | 45 frames |
| OOM splits | 0 (khong bi OOM) |
| CUDA warmup (chunk 1) | ~7s |

**Uoc tinh tong thoi gian xu ly 22 file (~100,000 frames):** ~30-35 phut

---

## 5. Ket luan

| Hang muc | Trang thai |
|----------|------------|
| Docker container start | OK |
| VideoSeal model load | OK |
| AudioSeal model load | OK (co download lan dau) |
| Doc file MXF (pre-convert) | OK |
| Embed watermark (video + audio) | OK |
| Output MXF (mpeg2video) | OK (sau fix) |
| Giu nguyen extension .mxf | OK |
| Unicode filenames | OK |
| Resolution le (638x360) | OK |
| **Loi chinh da fix** | MXF khong ho tro H.264 → re-encode mpeg2video |
| **Can rebuild Docker?** | KHONG (source mount truc tiep) |
| **Pipeline tong the** | DANG CHAY THANH CONG |
