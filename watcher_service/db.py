"""
SQLite database for the watcher service.
Tracks every job: pending / processing / done / error.
"""
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


_DB_PATH: Optional[str] = None


def init_db(db_path: str) -> None:
    global _DB_PATH
    _DB_PATH = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                input_path      TEXT    NOT NULL,
                filename        TEXT    NOT NULL,
                output_path     TEXT,
                watermark_text  TEXT,
                bits_list       TEXT,        -- JSON array of 0/1
                codeword_hex    TEXT,
                ecc_type        TEXT    DEFAULT 'bch',
                model_mode      TEXT    DEFAULT 'custom',
                center_mask     INTEGER DEFAULT 1,
                status          TEXT    DEFAULT 'pending',
                error_msg       TEXT,
                fps             REAL,
                has_audio       INTEGER DEFAULT 0,
                embed_time_s    REAL,
                created_at      TEXT,
                processed_at    TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_input  ON jobs(input_path)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)")


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(_DB_PATH, check_same_thread=False)


def reset_stale_jobs(max_age_minutes: int = 30) -> int:
    """Reset jobs stuck in 'processing' for longer than max_age_minutes.
    Called at service startup to recover from crashes / interrupted copies."""
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
    with _conn() as c:
        n = c.execute(
            "DELETE FROM jobs WHERE status='processing' AND created_at < ?",
            (cutoff,),
        ).rowcount
    return n


def is_processed(input_path: str) -> bool:
    """Return True if this path already has a job (processing, done, or error).
    Stale 'processing' jobs are cleaned up by reset_stale_jobs() at startup."""
    with _conn() as c:
        row = c.execute(
            "SELECT status FROM jobs WHERE input_path=? ORDER BY id DESC LIMIT 1",
            (input_path,),
        ).fetchone()
    return row is not None and row[0] in ("processing", "done", "error")


def insert_job(
    input_path: str,
    filename: str,
    watermark_text: str,
    bits_list: list,
    codeword_hex: str,
    ecc_type: str = "bch",
    model_mode: str = "custom",
    center_mask: bool = True,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        cur = c.execute(
            """INSERT INTO jobs
               (input_path, filename, watermark_text, bits_list, codeword_hex,
                ecc_type, model_mode, center_mask, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                input_path, filename, watermark_text,
                json.dumps(bits_list), codeword_hex,
                ecc_type, model_mode, int(center_mask),
                "processing", now,
            ),
        )
        return cur.lastrowid


def update_job_done(
    job_id: int,
    output_path: str,
    fps: float,
    embed_time_s: float,
    has_audio: bool = False,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            """UPDATE jobs
               SET status='done', output_path=?, fps=?,
                   embed_time_s=?, has_audio=?, processed_at=?
               WHERE id=?""",
            (output_path, fps, round(embed_time_s, 2), int(has_audio), now, job_id),
        )


def update_job_error(job_id: int, error_msg: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            "UPDATE jobs SET status='error', error_msg=?, processed_at=? WHERE id=?",
            (error_msg, now, job_id),
        )
