# Watcher Service — Optimization Progress

## Video: DJI 2688×1512 50fps HEVC → H.264 watermark embedding

### Baseline
- **34 f/s** (sequential pipeline, CPU-heavy)

### Optimization History

| # | Optimization | f/s | Speedup | Ghi chú |
|---|-------------|-----|---------|---------|
| 1 | Config fix: chunk_size 45→32, gpu_mem 0.9→0.7, emergency abort | — | — | Chống crash OOM |
| 2 | Fix logging: FileHandler ERROR→INFO, RotatingFileHandler 50MB×3 | — | — | Log ghi đúng file |
| 3 | Timing logs cho tất cả pipeline stages | — | — | Phát hiện bottleneck |
| 4 | Phân tích CPU vs GPU: encode ~55% thời gian | — | — | Bottleneck = encode |
| 5 | Breakdown encode: `from_ndarray` 76%, `encode` 23%, `mux` 1% | — | — | swscale RGB→YUV |
| 6 | GPU RGB→YUV420P (`_rgb_to_yuv420p_gpu`) | 77.7 | 2.3× | BT.709 matrix trên GPU |
| 7 | GPU resize_down trong `_embed_batch_fast` | 40.3 | — | DJI video test bắt đầu |
| 8 | GPU normalize uint8→float32 | 70.2 | +74% | Bỏ CPU normalize |
| 9 | FFmpeg subprocess pipe encode | 71.4 | +2% | Thay PyAV encode |
| 10 | YUV420P native decode (bypass swscale) + GPU convert | 80.7 | +13% | `_yuv420p_to_rgb_gpu` |
| 11 | Pinned memory + non_blocking GPU→CPU | 84.3 | +4.5% | Overlap transfer |
| 12 | GPU warmup (`_warmup_gpu`) | 112-115 | — | Fix first-chunk stutter |
| 13 | **Fix pinned buffer race condition** | **114.8** | — | **Fix nhảy hình frame 20-31** |

### Bug Fix #13 — Chi tiết

**Triệu chứng**: Video output bị "nhảy hình" trong 1-2 giây đầu. Frame 20-31 của chunk đầu tiên bị corrupt nghiêm trọng (PSNR rơi từ 37dB xuống 16dB).

**Nguyên nhân gốc**: Race condition trong 3-stage pipeline (`_embed_pipelined`):
- GPU thread dùng **1 pinned buffer** (`_pinned_buf`) để download kết quả từ GPU về CPU
- Sau download, tạo numpy **views** (không copy) vào `_pinned_buf` rồi đẩy vào `encode_q`
- GPU thread ngay lập tức xử lý chunk tiếp theo → **ghi đè `_pinned_buf`**
- Encode thread vẫn đang đọc views của chunk trước → **data bị corrupt**

**Timeline cụ thể** (từ log):
```
GPU chunk 1 xong: 56.175s → put views vào encode_q
GPU chunk 2 xong: 56.334s → GHI ĐÈ _pinned_buf
ENC chunk 1 xong: 56.531s → đọc data đã bị ghi đè (frame 20-31)
```

**Fix**: 
1. Double pinned buffer (`_pinned_bufs[0]`, `_pinned_bufs[1]`) — swap mỗi chunk
2. `.numpy().copy()` trước khi đẩy vào encode_q — tách ownership hoàn toàn

**Kết quả sau fix**:
- Frame 0-31 PSNR đồng đều ~37dB (trước: frame 20-31 rơi xuống 16dB)
- Chunk boundary Δ < 0.4dB (trước: 21dB jump)
- Không còn JUMP marker trong frame-to-frame MAE
- Tốc độ giữ nguyên 114.8 f/s

### Current Performance
- **114.8 f/s** (3.4× baseline)
- **32.5s** cho 3519 frames (70.4s video @ 50fps)
- Peak GPU memory: 2654MB
