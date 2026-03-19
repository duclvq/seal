"""
Phân tích service.log và hiển thị timing trực quan cho pipeline watermark.
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

LOG_FILE = Path(__file__).resolve().parent / "data" / "db" / "service.log"


def parse_log(log_path: str = None):
    path = Path(log_path) if log_path else LOG_FILE
    if not path.exists():
        print(f"Không tìm thấy log: {path}")
        return

    lines = path.read_text(encoding="utf-8").splitlines()

    # Tìm lần chạy cuối cùng (bắt đầu từ [EMBED-...)
    embed_start_idx = None
    for i, line in enumerate(lines):
        if "[EMBED-" in line:
            embed_start_idx = i
    if embed_start_idx is None:
        print("Không tìm thấy dữ liệu embedding trong log.")
        return

    run_lines = lines[embed_start_idx:]

    # Parse metadata
    meta = {}
    m = re.search(r"\[EMBED-([^\]]+)\]\s+(\S+):\s+(\d+x\d+),\s+~(\d+)f,\s+fps=([\d.]+),\s+chunk=(\d+)", run_lines[0])
    if m:
        meta["mode"] = m.group(1)
        meta["file"] = m.group(2)
        meta["res"] = m.group(3)
        meta["total_frames"] = int(m.group(4))
        meta["fps"] = float(m.group(5))
        meta["chunk"] = int(m.group(6))

    # Collect timing data
    gpu_times = []  # (n, normalize, embed, clamp, yuv, total)
    enc_times = []  # (n, from_ndarray, encode, mux, total, fps)
    dec_times = []  # (chunk#, decode, stack)
    delta_times = []  # (n, embedder, total)
    fast_times = []  # (n, resize_down, delta, blend, total)
    total_time = 0
    total_fps = 0

    for line in run_lines:
        # GPU timing
        mg = re.search(
            r"\[TIMING-PIPE-GPU\]\s+(\d+)f\s+"
            r"normalize=([\d.]+)s\s+embed=([\d.]+)s\s+"
            r"clamp\+uint8=([\d.]+)s\s+rgb2yuv_gpu=([\d.]+)s\s+"
            r"total=([\d.]+)s", line)
        if mg:
            gpu_times.append({
                "n": int(mg.group(1)), "normalize": float(mg.group(2)),
                "embed": float(mg.group(3)), "clamp": float(mg.group(4)),
                "yuv": float(mg.group(5)), "total": float(mg.group(6)),
            })

        # Encode timing
        me = re.search(
            r"\[TIMING-PIPE-ENC\]\s+(\d+)f\s+"
            r"from_ndarray=([\d.]+)s\s+encode=([\d.]+)s\s+mux=([\d.]+)s\s+"
            r"total=([\d.]+)s\s+\(([\d.]+)f/s\)", line)
        if me:
            enc_times.append({
                "n": int(me.group(1)), "from_ndarray": float(me.group(2)),
                "encode": float(me.group(3)), "mux": float(me.group(4)),
                "total": float(me.group(5)), "fps": float(me.group(6)),
            })

        # Decode timing
        md = re.search(
            r"\[TIMING-PIPE-DEC\]\s+chunk#(\d+)\s+"
            r"decode\+to_tensor=([\d.]+)s\s+stack\+pin=([\d.]+)s", line)
        if md:
            dec_times.append({
                "chunk": int(md.group(1)),
                "decode": float(md.group(2)), "stack": float(md.group(3)),
            })

        # Delta timing (embedder)
        mdt = re.search(
            r"\[TIMING-DELTA\]\s+(\d+)f\s+\S+\s+"
            r"upload=([\d.]+)s\s+embedder=([\d.]+)s.*?total=([\d.]+)s", line)
        if mdt:
            delta_times.append({
                "n": int(mdt.group(1)), "upload": float(mdt.group(2)),
                "embedder": float(mdt.group(3)), "total": float(mdt.group(4)),
            })

        # Fast timing
        mf = re.search(
            r"\[TIMING-FAST\]\s+(\d+)f\s+\S+\s+"
            r"resize_down=([\d.]+)s\s+delta=([\d.]+)s\s+"
            r"gpu_upload\+resize_up\+blend=([\d.]+)s\s+total=([\d.]+)s", line)
        if mf:
            fast_times.append({
                "n": int(mf.group(1)), "resize_down": float(mf.group(2)),
                "delta": float(mf.group(3)), "blend": float(mf.group(4)),
                "total": float(mf.group(5)),
            })

        # Total
        mt = re.search(r"video_embed=([\d.]+)s\s+\((\d+)f,\s+([\d.]+)f/s\)", line)
        if mt:
            total_time = float(mt.group(1))
            total_fps = float(mt.group(3))

    # ── Display ──────────────────────────────────────────────────────────
    W = 70  # bar width
    print()
    print("=" * 80)
    print(f"  PIPELINE TIMING ANALYSIS")
    print(f"  Mode: {meta.get('mode', '?')}")
    print(f"  File: {meta.get('file', '?')}")
    print(f"  Resolution: {meta.get('res', '?')}  Frames: {meta.get('total_frames', '?')}  FPS: {meta.get('fps', '?')}")
    print(f"  Chunk size: {meta.get('chunk', '?')}")
    print("=" * 80)

    # Skip first chunk (warmup)
    gpu_steady = gpu_times[1:] if len(gpu_times) > 1 else gpu_times
    enc_steady = enc_times[1:] if len(enc_times) > 1 else enc_times
    dec_steady = dec_times[1:] if len(dec_times) > 1 else dec_times
    delta_steady = delta_times[1:] if len(delta_times) > 1 else delta_times
    fast_steady = fast_times[1:] if len(fast_times) > 1 else fast_times

    def avg(lst, key):
        if not lst:
            return 0
        return sum(d[key] for d in lst) / len(lst)

    def bar(val, max_val, width=W, char="█"):
        if max_val <= 0:
            return ""
        n = int(val / max_val * width)
        return char * max(1, n)

    # ── 1. Overview per-chunk ────────────────────────────────────────────
    print()
    print("─" * 80)
    print("  TRUNG BÌNH MỖI CHUNK (bỏ chunk đầu warmup)")
    print("─" * 80)

    a_dec = avg(dec_steady, "decode") + avg(dec_steady, "stack")
    a_gpu = avg(gpu_steady, "total")
    a_enc = avg(enc_steady, "total")
    a_total_chunk = max(a_dec, a_gpu, a_enc)  # pipeline → bottleneck = max

    stages = [
        ("DECODE  (CPU)", a_dec, "🟦"),
        ("GPU     (embed+yuv)", a_gpu, "🟩"),
        ("ENCODE  (NVENC)", a_enc, "🟧"),
    ]

    max_stage = max(s[1] for s in stages) if stages else 1
    for name, val, icon in stages:
        b = bar(val, max_stage, W - 2)
        print(f"  {icon} {name:22s} {val*1000:6.0f}ms  {b}")

    print()
    print(f"  Pipeline bottleneck: {a_total_chunk*1000:.0f}ms/chunk → throughput ≈ {meta.get('chunk',30)/a_total_chunk:.0f} f/s (lý thuyết)")

    # ── 2. GPU breakdown ─────────────────────────────────────────────────
    print()
    print("─" * 80)
    print("  GPU STAGE BREAKDOWN (trung bình)")
    print("─" * 80)

    gpu_parts = [
        ("normalize (float/255)", avg(gpu_steady, "normalize"), "░"),
        ("embed (model inference)", avg(gpu_steady, "embed"), "█"),
        ("clamp+uint8", avg(gpu_steady, "clamp"), "▒"),
        ("rgb2yuv_gpu ★NEW", avg(gpu_steady, "yuv"), "▓"),
    ]
    max_gpu = max(p[1] for p in gpu_parts) if gpu_parts else 1
    for name, val, ch in gpu_parts:
        b = bar(val, max_gpu, W - 2, ch)
        print(f"    {name:28s} {val*1000:6.1f}ms  {b}")

    # ── 3. Encode breakdown ──────────────────────────────────────────────
    print()
    print("─" * 80)
    print("  ENCODE STAGE BREAKDOWN (trung bình)")
    print("─" * 80)

    enc_parts = [
        ("from_ndarray (YUV→AVFrame)", avg(enc_steady, "from_ndarray"), "░"),
        ("encode (NVENC H.264)", avg(enc_steady, "encode"), "█"),
        ("mux (write packet)", avg(enc_steady, "mux"), "▒"),
    ]
    max_enc = max(p[1] for p in enc_parts) if enc_parts else 1
    for name, val, ch in enc_parts:
        b = bar(val, max_enc, W - 2, ch)
        print(f"    {name:28s} {val*1000:6.1f}ms  {b}")

    avg_enc_fps = avg(enc_steady, "fps")
    print(f"\n    Encode throughput: {avg_enc_fps:.0f} f/s")

    # ── 4. Fast embed breakdown ──────────────────────────────────────────
    if fast_steady:
        print()
        print("─" * 80)
        print("  FAST EMBED BREAKDOWN (trung bình)")
        print("─" * 80)

        fast_parts = [
            ("resize_down (CPU→256)", avg(fast_steady, "resize_down"), "░"),
            ("delta (embedder 256²)", avg(fast_steady, "delta"), "█"),
            ("gpu_blend (upload+up+blend)", avg(fast_steady, "blend"), "▓"),
        ]
        max_fast = max(p[1] for p in fast_parts) if fast_parts else 1
        for name, val, ch in fast_parts:
            b = bar(val, max_fast, W - 2, ch)
            print(f"    {name:28s} {val*1000:6.1f}ms  {b}")

    # ── 5. Timeline visualization ────────────────────────────────────────
    print()
    print("─" * 80)
    print("  PIPELINE TIMELINE (mỗi dòng = 1 chunk, 3 stages chạy song song)")
    print("  D=Decode  G=GPU  E=Encode  (scale: 1 char ≈ 10ms)")
    print("─" * 80)

    SCALE = 0.010  # 10ms per char
    n_show = min(10, len(gpu_times))  # show first 10 chunks
    for i in range(n_show):
        d_len = 0
        if i < len(dec_times):
            d_len = int((dec_times[i]["decode"] + dec_times[i]["stack"]) / SCALE)
        g_len = int(gpu_times[i]["total"] / SCALE) if i < len(gpu_times) else 0
        e_len = int(enc_times[i]["total"] / SCALE) if i < len(enc_times) else 0

        d_bar = "D" * max(1, d_len)
        g_bar = "G" * max(1, g_len)
        e_bar = "E" * max(1, e_len)

        chunk_label = f"  #{i+1:2d}"
        print(f"{chunk_label}  {d_bar} {g_bar} {e_bar}")

    if len(gpu_times) > n_show:
        print(f"  ... ({len(gpu_times) - n_show} chunks nữa)")

    # ── 6. Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"  TỔNG KẾT")
    print(f"  ├─ Tổng thời gian embed: {total_time:.1f}s")
    print(f"  ├─ Throughput thực tế:    {total_fps:.1f} f/s")
    print(f"  ├─ Số chunks:            {len(gpu_times)}")
    print(f"  ├─ GPU avg/chunk:        {avg(gpu_steady, 'total')*1000:.0f}ms")
    print(f"  │   ├─ embed:            {avg(gpu_steady, 'embed')*1000:.0f}ms")
    print(f"  │   └─ rgb2yuv_gpu:      {avg(gpu_steady, 'yuv')*1000:.0f}ms")
    print(f"  ├─ Encode avg/chunk:     {avg(enc_steady, 'total')*1000:.0f}ms")
    print(f"  │   ├─ from_ndarray:     {avg(enc_steady, 'from_ndarray')*1000:.0f}ms (was ~650ms before GPU YUV)")
    print(f"  │   └─ NVENC encode:     {avg(enc_steady, 'encode')*1000:.0f}ms")
    print(f"  └─ Decode avg/chunk:     {a_dec*1000:.0f}ms")
    print("=" * 80)
    print()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else None
    parse_log(log_path)
