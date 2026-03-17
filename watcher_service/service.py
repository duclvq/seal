"""
VideoSeal Watcher Service
=========================
Watches INPUT_FOLDER for new video files, embeds watermarks (video + audio),
saves to OUTPUT_FOLDER, and persists results to SQLite.

Config via .env (or environment variables):
    INPUT_FOLDER        — folder to watch  (default /data/input)
    OUTPUT_FOLDER       — output folder    (default /data/output)
    DB_PATH             — SQLite DB path   (default /data/watermarks.db)
    ERROR_LOG           — error log path   (default /data/error.log)
    WATERMARK_TEXT      — fixed text to embed; empty = random UUID per file
    CUSTOM_CKPT_PATH    — path to custom VideoSeal checkpoint
    DEVICE              — "cuda" | "cpu"   (default "cuda")
    SCAN_EXISTING       — "true" | "false" (default "true")
    CHUNK_SIZE          — frames per GPU batch (default 30, tăng để dùng nhiều VRAM hơn)
    GPU_MEMORY_FRACTION — fraction of VRAM allowed [0.1–1.0] (default 0.9)
    CUDA_VISIBLE_DEVICES— GPU index to use, e.g. "0" or "0,1" (default: all)
"""

import csv
import logging
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path

# ── Load .env before anything else ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass   # python-dotenv optional; env vars may already be set

# ── Project root on sys.path ─────────────────────────────────────────────────
_SERVICE_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_FOLDER     = Path(os.getenv("INPUT_FOLDER",    "/data/input"))
OUTPUT_FOLDER    = Path(os.getenv("OUTPUT_FOLDER",   "/data/output"))
DB_PATH          = Path(os.getenv("DB_PATH",         "/data/watermarks.db"))
ERROR_LOG        = Path(os.getenv("ERROR_LOG",       "/data/error.log"))
WATERMARK_TEXT   = os.getenv("WATERMARK_TEXT", "").strip()
CUSTOM_CKPT_PATH = os.getenv("CUSTOM_CKPT_PATH", "")
DEVICE_STR           = os.getenv("DEVICE", "cuda")
SCAN_EXISTING        = os.getenv("SCAN_EXISTING", "true").lower() == "true"
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE", "30"))
GPU_MEMORY_FRACTION  = float(os.getenv("GPU_MEMORY_FRACTION", "0.9"))
NUM_WORKERS          = int(os.getenv("NUM_WORKERS", "1"))
STATS_CSV            = Path(os.getenv("STATS_CSV").strip()) if os.getenv("STATS_CSV", "").strip() else None

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".ts", ".flv", ".mxf"}

# ── Logging ───────────────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)

    fh = logging.FileHandler(str(ERROR_LOG), encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    return logging.getLogger(__name__)

log = _setup_logging()

# ── Job queue (single worker = single GPU stream) ────────────────────────────
_job_queue: queue.Queue = queue.Queue()
_queued_paths: set      = set()           # dedup: paths currently in queue
_queued_lock            = threading.Lock()
_stop_event             = threading.Event()
_csv_lock               = threading.Lock()

# ── Stats CSV ────────────────────────────────────────────────────────────────
_STATS_HEADER = [
    "timestamp", "filename", "resolution", "total_frames", "fps",
    "embed_time_s", "throughput_fps", "has_audio",
    "gpu_alloc_mb", "gpu_reserved_mb", "gpu_peak_mb",
    "chunk_size", "num_oom_splits", "status",
]

def _init_stats_csv():
    if STATS_CSV is None:
        return
    p = Path(STATS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_STATS_HEADER)
    log.info(f"Stats CSV: {p}")

def _write_stats_row(row: dict):
    if STATS_CSV is None:
        return
    with _csv_lock:
        with open(STATS_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_STATS_HEADER, extrasaction="ignore")
            w.writerow(row)


# ── Worker thread ─────────────────────────────────────────────────────────────
def _worker(video_model, audio_model, device):
    import torch
    from db     import insert_job, update_job_done, update_job_error
    from worker import embed_to_file
    from web_demo.core.ecc import text_to_msg_tensor_bch, bch_info

    MSG_BITS = 256

    while not _stop_event.is_set():
        try:
            input_path: Path = _job_queue.get(timeout=1)
        except queue.Empty:
            continue

        filename = input_path.name
        # Determine watermark text (BCH allows max 4 ASCII bytes)
        wm_text = WATERMARK_TEXT[:4] if WATERMARK_TEXT else os.urandom(2).hex()  # e.g. "a3f2"

        # Pre-encode to get bits_list for DB record
        try:
            _msg_tensor, codeword, bits_list = text_to_msg_tensor_bch(wm_text, msg_bits=MSG_BITS)
        except Exception as e:
            log.error(f"[ECC] {filename}: {e}", exc_info=True)
            with _queued_lock:
                _queued_paths.discard(str(input_path))
            _job_queue.task_done()
            continue

        job_id = insert_job(
            input_path    = str(input_path),
            filename      = filename,
            watermark_text= wm_text,
            bits_list     = bits_list,
            codeword_hex  = codeword.hex(),
            ecc_type      = "bch",
            model_mode    = "custom",
            center_mask   = True,
        )

        # Preserve subfolder structure: input/a/b/c.mxf → output/a/b/c.mxf
        try:
            rel = input_path.relative_to(INPUT_FOLDER)
        except ValueError:
            rel = Path(filename)
        output_path = OUTPUT_FOLDER / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            result = embed_to_file(
                input_path   = str(input_path),
                output_path  = str(output_path),
                video_model  = video_model,
                watermark_text = wm_text,
                device       = device,
                audio_model  = audio_model,
            )
            embed_time = time.time() - t0
            fps       = result["fps"]
            n_frames  = result["total_frames"]
            has_audio = result["has_audio"]
            res       = result["resolution"]
            oom_splits = result["oom_splits"]

            update_job_done(job_id, str(output_path), fps, embed_time, has_audio)

            # GPU stats
            gpu_alloc = gpu_resv = gpu_peak = 0.0
            gpu_info = ""
            if device.type == "cuda":
                gpu_alloc = torch.cuda.memory_allocated(device) / 1024**2
                gpu_resv  = torch.cuda.memory_reserved(device) / 1024**2
                gpu_peak  = torch.cuda.max_memory_allocated(device) / 1024**2
                gpu_info = f"  GPU: {gpu_alloc:.0f}/{gpu_resv:.0f}MB (peak {gpu_peak:.0f}MB)"
                torch.cuda.reset_peak_memory_stats(device)

            fps_proc = n_frames / embed_time if embed_time > 0 else 0
            oom_s = f"  OOM-splits={oom_splits}" if oom_splits else ""
            log.info(
                f"[DONE] {filename} [{res}]  →  {output_path.name}"
                f"  ({embed_time:.1f}s, {n_frames}f, {fps_proc:.1f}f/s, audio={'yes' if has_audio else 'no'}){gpu_info}{oom_s}"
            )

            # Write stats CSV row
            _write_stats_row({
                "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename":       filename,
                "resolution":     res,
                "total_frames":   n_frames,
                "fps":            round(fps, 2),
                "embed_time_s":   round(embed_time, 2),
                "throughput_fps": round(fps_proc, 2),
                "has_audio":      has_audio,
                "gpu_alloc_mb":   round(gpu_alloc),
                "gpu_reserved_mb": round(gpu_resv),
                "gpu_peak_mb":    round(gpu_peak),
                "chunk_size":     CHUNK_SIZE,
                "num_oom_splits": oom_splits,
                "status":         "done",
            })

        except Exception as exc:
            update_job_error(job_id, str(exc))
            log.error(f"[FAIL] {filename}: {exc}", exc_info=True)
            _write_stats_row({
                "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
                "filename":  filename,
                "status":    "error",
            })
        finally:
            with _queued_lock:
                _queued_paths.discard(str(input_path))
            if device.type == "cuda":
                import torch as _torch
                _torch.cuda.empty_cache()
            _job_queue.task_done()


# ── Enqueue helper ────────────────────────────────────────────────────────────
def _is_file_stable(path: Path, wait_seconds: int = 3) -> bool:
    """Return True if the file size hasn't changed over wait_seconds.
    Prevents processing files that are still being copied."""
    import time
    try:
        size1 = path.stat().st_size
        if size1 == 0:
            return False
        time.sleep(wait_seconds)
        size2 = path.stat().st_size
        return size1 == size2
    except OSError:
        return False


def _is_valid_video(path: Path) -> bool:
    """Quick check: can PyAV open the file and find a video stream?"""
    try:
        import av
        with av.open(str(path)) as c:
            return len(c.streams.video) > 0
    except Exception:
        return False


def _enqueue(path: Path) -> None:
    from db import is_processed
    if not path.is_file():
        return
    if path.name.startswith("._"):
        return  # macOS resource fork / AppleDouble metadata file
    if path.suffix.lower() not in ALLOWED_EXT:
        return
    with _queued_lock:
        if str(path) in _queued_paths:
            return  # already in queue, skip
    if is_processed(str(path)):
        log.debug(f"[SKIP] already done: {path.name}")
        return
    if not _is_file_stable(path):
        log.info(f"[WAIT] file still changing, skip for now: {path.name}")
        return
    if not _is_valid_video(path):
        log.warning(f"[SKIP] invalid/corrupt video: {path.name}")
        return
    with _queued_lock:
        if str(path) in _queued_paths:
            return  # re-check after slow validation
        _queued_paths.add(str(path))
    log.info(f"[QUEUE] {path.name}")
    _job_queue.put(path)


# ── Watchdog event handler ────────────────────────────────────────────────────
class _VideoHandler:
    """Minimal watchdog FileSystemEventHandler."""

    def dispatch(self, event):
        if event.event_type in ("created", "moved"):
            self._handle(event)

    def _handle(self, event):
        if getattr(event, "is_directory", False):
            return
        src = Path(getattr(event, "dest_path", None) or event.src_path)
        if src.suffix.lower() not in ALLOWED_EXT:
            return
        # Brief pause so large files finish copying before we open them
        time.sleep(1.5)
        _enqueue(src)


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # DB
    from db import init_db, reset_stale_jobs
    init_db(str(DB_PATH))
    stale = reset_stale_jobs(max_age_minutes=30)
    if stale:
        log.info(f"Reset {stale} stale processing job(s) — will retry")
    log.info(f"DB: {DB_PATH}")

    # Stats CSV
    _init_stats_csv()

    # Device
    import torch
    if DEVICE_STR == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(DEVICE_STR)

    if device.type == "cuda":
        frac = max(0.1, min(1.0, GPU_MEMORY_FRACTION))
        device_idx = device.index if device.index is not None else 0
        torch.cuda.set_per_process_memory_fraction(frac, device_idx)
        total_mb = torch.cuda.get_device_properties(device_idx).total_memory / 1024**2
        log.info(f"Device: {device}  |  VRAM limit: {frac*100:.0f}%  ({frac*total_mb:.0f} / {total_mb:.0f} MB)")
    else:
        log.info(f"Device: {device}")

    # Load models (once, before watchdog starts)
    if not CUSTOM_CKPT_PATH:
        log.error("CUSTOM_CKPT_PATH is not set in .env — cannot start.")
        sys.exit(1)
    ckpt = Path(CUSTOM_CKPT_PATH)
    if not ckpt.is_absolute():
        ckpt = _PROJECT_ROOT / ckpt
    if not ckpt.is_file():
        log.error(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    log.info(f"Loading VideoSeal model from {ckpt} ...")
    from worker import load_video_model, load_audio_model
    video_model = load_video_model(str(ckpt), device)
    log.info("VideoSeal model ready.")

    log.info("Loading AudioSeal model ...")
    audio_model = load_audio_model(device)
    if audio_model is None:
        log.warning("AudioSeal unavailable — audio watermark will be skipped.")
    else:
        log.info("AudioSeal model ready.")

    # Worker threads
    n_workers = max(1, NUM_WORKERS)
    log.info(f"Starting {n_workers} worker thread(s) ...")
    for i in range(n_workers):
        t = threading.Thread(
            target=_worker,
            args=(video_model, audio_model, device),
            daemon=True,
            name=f"watcher-worker-{i}",
        )
        t.start()

    # Scan existing files first
    if SCAN_EXISTING:
        log.info("Scanning existing files ...")
        for p in sorted(INPUT_FOLDER.rglob("*")):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                _enqueue(p)

    # Watchdog observer — use PollingObserver for Docker/network mounts
    try:
        from watchdog.observers.polling import PollingObserver
        observer = PollingObserver(timeout=10)
        obs_type = "polling"
    except ImportError:
        from watchdog.observers import Observer
        observer = Observer()
        obs_type = "inotify"

    from watchdog.events import FileSystemEventHandler

    class _Handler(FileSystemEventHandler):
        def on_created(self, event):
            _VideoHandler().dispatch(event)
        def on_moved(self, event):
            _VideoHandler().dispatch(event)

    observer.schedule(_Handler(), str(INPUT_FOLDER), recursive=True)
    observer.start()
    log.info(f"Watching ({obs_type}): {INPUT_FOLDER}  →  OUTPUT: {OUTPUT_FOLDER}")

    # Periodic rescan — catches files watchdog might miss (Docker bind mounts, NFS)
    POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))  # seconds, 0 = disable

    def _poll_loop():
        while not _stop_event.is_set():
            _stop_event.wait(POLL_INTERVAL)
            if _stop_event.is_set():
                break
            for p in sorted(INPUT_FOLDER.rglob("*")):
                if _stop_event.is_set():
                    break
                if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                    _enqueue(p)

    if POLL_INTERVAL > 0:
        poll_thread = threading.Thread(target=_poll_loop, daemon=True, name="poll-rescan")
        poll_thread.start()
        log.info(f"Periodic rescan every {POLL_INTERVAL}s")

    # Graceful shutdown on SIGTERM / SIGINT
    def _shutdown(signum, frame):
        log.info("Shutdown signal received, stopping ...")
        _stop_event.set()
        observer.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    try:
        while not _stop_event.is_set():
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
        _job_queue.join()    # wait for in-flight jobs
        log.info("Watcher service stopped.")


if __name__ == "__main__":
    main()
