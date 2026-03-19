"""
VideoSeal Watcher Service
=========================
Watches INPUT_FOLDER for new video files, embeds watermarks (video + audio),
saves to OUTPUT_FOLDER, and persists results to SQLite.

Config via .env (or environment variables), overridden by config.yaml:
    INPUT_FOLDER        — folder to watch  (default /data/input)
    OUTPUT_FOLDER       — output folder    (default /data/output)
    DB_PATH             — SQLite DB path   (default /data/watermarks.db)
    ERROR_LOG           — error log path   (default /data/error.log)
    WATERMARK_TEXT      — fixed text to embed; empty = random UUID per file
    CUSTOM_CKPT_PATH    — path to custom VideoSeal checkpoint
    DEVICE              — "cuda" | "cpu"   (default "cuda")
    SCAN_EXISTING       — "true" | "false" (default "true")
    CHUNK_SIZE          — frames per GPU batch (default 30)
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

import yaml

# ── Load .env before anything else ───────────────────────────────────────────
# .env_home overrides .env (for local/home dev machine)
try:
    from dotenv import load_dotenv
    _env_home = Path(__file__).parent / ".env_home"
    _env_base = Path(__file__).parent / ".env"
    if _env_home.exists():
        load_dotenv(_env_home, override=True)
        print(f"[CONFIG] Loaded .env_home (overrides .env)")
    elif _env_base.exists():
        load_dotenv(_env_base)
except ImportError:
    pass   # python-dotenv optional; env vars may already be set

# ── Project root on sys.path ─────────────────────────────────────────────────
_SERVICE_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

# ── Config (from .env) ──────────────────────────────────────────────────────
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
FAST_EMBED           = os.getenv("FAST_EMBED", "true").lower() == "true"
PIPELINE             = os.getenv("PIPELINE", "true").lower() == "true"
STATS_CSV            = Path(os.getenv("STATS_CSV").strip()) if os.getenv("STATS_CSV", "").strip() else None

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".ts", ".flv", ".mxf"}

# ── Memory safety limits ─────────────────────────────────────────────────────
RAM_LIMIT_PCT       = float(os.getenv("RAM_LIMIT_PCT", "75"))        # max % system RAM
VRAM_LIMIT_PCT      = float(os.getenv("VRAM_LIMIT_PCT", "85"))       # max % VRAM
MEMORY_CHECK_INTERVAL = int(os.getenv("MEMORY_CHECK_INTERVAL", "5")) # seconds

# ── config.yaml override ────────────────────────────────────────────────────
CONFIG_YAML = _SERVICE_DIR / "config.yaml"
_config_mtime: float = 0.0


def _load_config_yaml(initial: bool = False) -> bool:
    """Load config.yaml and override .env values.
    Returns True if service restart is requested."""
    global WATERMARK_TEXT, CHUNK_SIZE, SCAN_EXISTING, FAST_EMBED, _config_mtime

    if not CONFIG_YAML.exists():
        return False

    try:
        mtime = CONFIG_YAML.stat().st_mtime
    except OSError:
        return False

    if not initial and mtime == _config_mtime:
        return False  # no change

    _config_mtime = mtime

    try:
        with open(CONFIG_YAML, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        if initial:
            print(f"[WARN] Failed to read config.yaml: {e}")
        else:
            log.warning(f"Failed to read config.yaml: {e}")
        return False

    changed = []

    if "watermark_text" in cfg:
        new_wm = str(cfg["watermark_text"]).strip()
        if new_wm != WATERMARK_TEXT:
            changed.append(f"watermark_text: '{WATERMARK_TEXT}' -> '{new_wm}'")
            WATERMARK_TEXT = new_wm

    if "chunk_size" in cfg:
        new_cs = int(cfg["chunk_size"])
        if new_cs != CHUNK_SIZE:
            changed.append(f"chunk_size: {CHUNK_SIZE} -> {new_cs}")
            CHUNK_SIZE = new_cs
            try:
                import worker
                worker.CHUNK_SIZE = new_cs
            except ImportError:
                pass

    if "scan_existing" in cfg:
        new_se = bool(cfg["scan_existing"])
        if new_se != SCAN_EXISTING:
            changed.append(f"scan_existing: {SCAN_EXISTING} -> {new_se}")
            SCAN_EXISTING = new_se

    if "fast_embed" in cfg:
        new_fe = bool(cfg["fast_embed"])
        if new_fe != FAST_EMBED:
            changed.append(f"fast_embed: {FAST_EMBED} -> {new_fe}")
            FAST_EMBED = new_fe
            try:
                import worker
                worker.FAST_EMBED = new_fe
            except ImportError:
                pass

    if changed and not initial:
        for c in changed:
            log.info(f"[CONFIG] {c}")

    # Check restart flag
    if cfg.get("restart"):
        if not initial:
            log.info("[CONFIG] Restart requested via config.yaml")
        # Clear the flag
        cfg["restart"] = False
        try:
            with open(CONFIG_YAML, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        except Exception:
            pass
        return True

    return False


# Apply config.yaml overrides at import time (before logging setup)
_restart_at_start = _load_config_yaml(initial=True)

# ── Logging ───────────────────────────────────────────────────────────────────
# Log files:
#   - error.log   : chỉ ERROR+CRITICAL (như cũ)
#   - service.log : toàn bộ INFO+ (timing, progress, memory, v.v.)
# Cả hai nằm cùng thư mục với ERROR_LOG (DATA_DIR/db/).

def _setup_logging() -> logging.Logger:
    from logging.handlers import RotatingFileHandler

    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)-8s] %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console (stdout)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(sh)

    # Error-only file (error.log)
    fh = logging.FileHandler(str(ERROR_LOG), encoding="utf-8")
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)

    # Full log file (service.log) — INFO+, rotating 50MB x 3 backups
    full_log_path = ERROR_LOG.parent / "service.log"
    fh_full = RotatingFileHandler(
        str(full_log_path), maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh_full.setLevel(logging.INFO)
    fh_full.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh_full)

    return logging.getLogger(__name__)

log = _setup_logging()

# ── Job queue (single worker = single GPU stream) ────────────────────────────
_job_queue: queue.Queue = queue.Queue()
_queued_paths: set      = set()           # dedup: paths currently in queue
_queued_lock            = threading.Lock()
_stop_event             = threading.Event()
_memory_emergency       = threading.Event()  # set when memory is critically high
_csv_lock               = threading.Lock()

# ── Stats CSV ────────────────────────────────────────────────────────────────
_STATS_HEADER = [
    "timestamp", "filename", "resolution", "total_frames", "fps",
    "embed_time_s", "throughput_fps", "has_audio",
    "gpu_alloc_mb", "gpu_reserved_mb", "gpu_peak_mb",
    "chunk_size", "num_oom_splits", "status",
]


# ── Memory safety monitor ────────────────────────────────────────────────────
def _check_memory_safe() -> tuple[bool, str]:
    """Check if RAM and VRAM usage are within safe limits.
    Returns (is_safe, reason_if_not_safe)."""
    import psutil

    # RAM check
    ram = psutil.virtual_memory()
    ram_pct = ram.percent
    if ram_pct > RAM_LIMIT_PCT:
        return False, f"RAM {ram_pct:.0f}% > limit {RAM_LIMIT_PCT:.0f}% ({ram.used/1024**3:.1f}/{ram.total/1024**3:.1f} GB)"

    # VRAM check
    if DEVICE_STR == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = 0
                total = torch.cuda.get_device_properties(device_idx).total_memory
                allocated = torch.cuda.memory_allocated(device_idx)
                reserved = torch.cuda.memory_reserved(device_idx)
                # Use reserved (actual GPU allocation) as the metric
                vram_pct = reserved / total * 100
                if vram_pct > VRAM_LIMIT_PCT:
                    return False, (f"VRAM {vram_pct:.0f}% > limit {VRAM_LIMIT_PCT:.0f}% "
                                   f"(alloc={allocated/1024**2:.0f}MB reserved={reserved/1024**2:.0f}MB "
                                   f"total={total/1024**2:.0f}MB)")
        except Exception:
            pass  # If we can't check VRAM, don't block

    return True, ""


def _memory_monitor_loop():
    """Background thread: periodically check memory and pause queue if needed."""
    import psutil
    consecutive_warnings = 0

    while not _stop_event.is_set():
        _stop_event.wait(MEMORY_CHECK_INTERVAL)
        if _stop_event.is_set():
            break

        is_safe, reason = _check_memory_safe()
        if not is_safe:
            consecutive_warnings += 1
            log.warning(f"[MEMORY] {reason} (warning #{consecutive_warnings})")

            # After 3 consecutive warnings, force garbage collection
            if consecutive_warnings >= 3:
                log.warning("[MEMORY] Forcing garbage collection + CUDA cache clear")
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

            # After 6 consecutive warnings, log critical (service should be restarted)
            if consecutive_warnings >= 6:
                log.critical(
                    f"[MEMORY] Sustained high memory for {consecutive_warnings * MEMORY_CHECK_INTERVAL}s. "
                    f"Consider reducing CHUNK_SIZE or NUM_WORKERS."
                )

            # EMERGENCY: After 12 consecutive warnings (~60s), set pause flag
            # to signal workers to abort current job
            if consecutive_warnings >= 12:
                _memory_emergency.set()
                log.critical("[MEMORY] EMERGENCY — signaling workers to abort current job")
        else:
            if consecutive_warnings > 0:
                log.info(f"[MEMORY] Back to safe levels after {consecutive_warnings} warnings")
            consecutive_warnings = 0
            _memory_emergency.clear()

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
    from db     import update_job_start, update_job_done, update_job_error, update_job_progress
    from worker import embed_to_file
    from web_demo.core.ecc import text_to_msg_tensor_bch, bch_info

    MSG_BITS = 256

    while not _stop_event.is_set():
        try:
            job_id, input_path = _job_queue.get(timeout=1)
        except queue.Empty:
            continue

        filename = input_path.name
        # ── Memory safety gate: wait if memory is too high ────────────────
        _mem_wait_count = 0
        while not _stop_event.is_set():
            is_safe, reason = _check_memory_safe()
            if is_safe:
                break
            _mem_wait_count += 1
            if _mem_wait_count == 1:
                log.warning(f"[MEMORY-GATE] {filename}: pausing — {reason}")
            # Force cleanup
            import gc
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _stop_event.wait(10)  # wait 10s before re-checking
        if _mem_wait_count > 0:
            log.info(f"[MEMORY-GATE] {filename}: resumed after {_mem_wait_count * 10}s wait")
        if _stop_event.is_set():
            with _queued_lock:
                _queued_paths.discard(str(input_path))
            _job_queue.task_done()
            continue

        # Determine watermark text (BCH allows max 4 ASCII bytes)
        wm_text = WATERMARK_TEXT[:4] if WATERMARK_TEXT else os.urandom(2).hex()  # e.g. "a3f2"

        # Pre-encode to get bits_list for DB record
        try:
            _msg_tensor, codeword, bits_list = text_to_msg_tensor_bch(wm_text, msg_bits=MSG_BITS)
        except Exception as e:
            log.error(f"[ECC] {filename}: {e}", exc_info=True)
            update_job_error(job_id, str(e))
            with _queued_lock:
                _queued_paths.discard(str(input_path))
            _job_queue.task_done()
            continue

        # Transition pending → processing with watermark details
        update_job_start(
            job_id,
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

        # Progress callback — updates DB after each chunk
        def _progress_cb(processed, total, _jid=job_id):
            try:
                update_job_progress(_jid, processed, total)
            except Exception:
                pass  # non-critical

        t0 = time.time()
        try:
            result = embed_to_file(
                input_path   = str(input_path),
                output_path  = str(output_path),
                video_model  = video_model,
                watermark_text = wm_text,
                device       = device,
                audio_model  = audio_model,
                progress_callback = _progress_cb,
            )
            embed_time = time.time() - t0
            fps       = result["fps"]
            n_frames  = result["total_frames"]
            has_audio = result["has_audio"]
            res       = result["resolution"]
            oom_splits = result["oom_splits"]

            update_job_done(
                job_id, str(output_path), fps, embed_time, has_audio,
                resolution=res, total_frames=n_frames,
            )

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
    from db import is_processed, insert_pending_job
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
    # Insert pending record into DB so monitor can see the queue
    job_id = insert_pending_job(str(path), path.name)
    log.info(f"[QUEUE] {path.name} (job #{job_id})")
    _job_queue.put((job_id, path))


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

    # Log config.yaml status
    if CONFIG_YAML.exists():
        log.info(f"Config YAML: {CONFIG_YAML} (active, hot-reload enabled)")
    else:
        log.info(f"Config YAML: {CONFIG_YAML} (not found, using .env only)")

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

    # Memory monitor thread
    mem_thread = threading.Thread(target=_memory_monitor_loop, daemon=True, name="memory-monitor")
    mem_thread.start()
    log.info(f"Memory monitor: RAM limit={RAM_LIMIT_PCT:.0f}%, VRAM limit={VRAM_LIMIT_PCT:.0f}%, "
             f"check every {MEMORY_CHECK_INTERVAL}s")

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
            # Hot-reload config.yaml
            if _load_config_yaml():
                log.info("[CONFIG] Restart flag detected — stopping service ...")
                _stop_event.set()
                break
            for p in sorted(INPUT_FOLDER.rglob("*")):
                if _stop_event.is_set():
                    break
                if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                    _enqueue(p)

    if POLL_INTERVAL > 0:
        poll_thread = threading.Thread(target=_poll_loop, daemon=True, name="poll-rescan")
        poll_thread.start()
        log.info(f"Periodic rescan every {POLL_INTERVAL}s (+ config.yaml hot-reload)")

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
