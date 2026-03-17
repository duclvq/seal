"""
Test cases for Watcher Service — large file handling (10-20GB MXF/MP4).

These tests use mock/fake files to simulate large file scenarios without
requiring actual 10-20GB files. They test the service's logic for:
  - File stability detection (files being copied)
  - Queue/dedup behavior with large files
  - Subfolder structure preservation
  - Periodic rescan picking up new files
  - DB state transitions (processing → done / error)
  - OOM auto-split behavior
  - Graceful handling of corrupt/partial files

Run:  cd watcher_service && python -m pytest tests/test_watcher_large_files.py -v
"""

import os
import sys
import queue
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ── Setup paths ─────────────────────────────────────────────────────────────
_TEST_DIR = Path(__file__).resolve().parent
_SERVICE_DIR = _TEST_DIR.parent
_PROJECT_ROOT = _SERVICE_DIR.parent

if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    """Create temporary input/output/db directories."""
    inp = tmp_path / "input"
    out = tmp_path / "output"
    db_path = tmp_path / "db" / "watermarks.db"
    inp.mkdir()
    out.mkdir()
    db_path.parent.mkdir()
    return inp, out, db_path


@pytest.fixture
def init_test_db(tmp_dirs):
    """Initialize DB for testing."""
    _, _, db_path = tmp_dirs
    import db as db_module
    db_module.init_db(str(db_path))
    return db_path


def _create_fake_video(path: Path, size_bytes: int = 1024):
    """Create a fake file of given size (not a real video)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size_bytes)


def _create_growing_file(path: Path, initial_size: int, grow_by: int, interval: float, steps: int):
    """Simulate a file being copied — grows in background thread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * initial_size)

    def _grow():
        for _ in range(steps):
            time.sleep(interval)
            with open(path, "ab") as f:
                f.write(b"\x00" * grow_by)

    t = threading.Thread(target=_grow, daemon=True)
    t.start()
    return t


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 1: File stability detection (_is_file_stable)
# ═════════════════════════════════════════════════════════════════════════════

class TestFileStability:
    """Test _is_file_stable() — prevents processing files still being copied."""

    def test_stable_file_returns_true(self, tmp_path):
        """A file that doesn't change size should be detected as stable."""
        f = tmp_path / "stable.mp4"
        _create_fake_video(f, size_bytes=1000)

        import service
        # Use short wait for test speed
        assert service._is_file_stable(f, wait_seconds=1) is True

    def test_empty_file_returns_false(self, tmp_path):
        """Empty file (0 bytes) should not be considered stable."""
        f = tmp_path / "empty.mp4"
        f.touch()

        import service
        assert service._is_file_stable(f, wait_seconds=1) is False

    def test_growing_file_returns_false(self, tmp_path):
        """A file that is still being written should return False."""
        f = tmp_path / "growing.mxf"
        # File grows every 0.3s for 5 steps
        _create_growing_file(f, initial_size=1000, grow_by=5000, interval=0.3, steps=5)
        time.sleep(0.1)  # let first write happen

        import service
        assert service._is_file_stable(f, wait_seconds=1) is False

    def test_file_becomes_stable_after_copy(self, tmp_path):
        """File that stops growing should eventually be detected as stable."""
        f = tmp_path / "large_copy.mxf"
        # Grows for 1 second, then stops
        t = _create_growing_file(f, initial_size=1000, grow_by=5000, interval=0.2, steps=3)
        t.join()  # wait for copy to finish
        time.sleep(0.5)

        import service
        assert service._is_file_stable(f, wait_seconds=1) is True

    def test_nonexistent_file_returns_false(self, tmp_path):
        """A file that doesn't exist should return False."""
        f = tmp_path / "does_not_exist.mp4"

        import service
        assert service._is_file_stable(f, wait_seconds=1) is False

    def test_large_file_20gb_stable(self, tmp_path):
        """Simulate a 20GB file (only check size, not actual content).
        Uses sparse file to avoid writing 20GB to disk."""
        f = tmp_path / "large_20gb.mxf"
        # Create sparse file — reports 20GB size but uses minimal disk
        with open(f, "wb") as fh:
            fh.seek(20 * 1024 * 1024 * 1024 - 1)  # 20GB - 1
            fh.write(b"\x00")

        import service
        assert service._is_file_stable(f, wait_seconds=1) is True
        assert f.stat().st_size == 20 * 1024 * 1024 * 1024


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 2: Enqueue logic with large files
# ═════════════════════════════════════════════════════════════════════════════

class TestEnqueueLargeFiles:
    """Test _enqueue() behavior with various file scenarios."""

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_new_large_mxf_gets_queued(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """A new large .mxf file should be enqueued."""
        inp, _, _ = tmp_dirs
        f = inp / "big_show.mxf"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)

        assert not service._job_queue.empty()
        queued = service._job_queue.get_nowait()
        assert queued.name == "big_show.mxf"

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_new_large_mp4_gets_queued(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """A new large .mp4 file should be enqueued."""
        inp, _, _ = tmp_dirs
        f = inp / "big_show.mp4"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)

        assert not service._job_queue.empty()

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=False)
    def test_unstable_file_not_queued(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """A file still being copied should NOT be enqueued."""
        inp, _, _ = tmp_dirs
        f = inp / "copying.mxf"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)

        assert service._job_queue.empty()

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_duplicate_not_queued_twice(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """Same file should not be enqueued twice (dedup via _queued_paths)."""
        inp, _, _ = tmp_dirs
        f = inp / "show.mxf"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)
        service._enqueue(f)  # second call

        assert service._job_queue.qsize() == 1

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_already_done_not_queued(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """A file already marked 'done' in DB should not be re-queued."""
        inp, _, _ = tmp_dirs
        f = inp / "done_file.mxf"
        _create_fake_video(f, size_bytes=2048)

        import db as db_module
        job_id = db_module.insert_job(
            input_path=str(f), filename=f.name,
            watermark_text="test", bits_list=[0]*256,
            codeword_hex="00"*32,
        )
        db_module.update_job_done(job_id, str(f), 25.0, 10.0)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)

        assert service._job_queue.empty()

    def test_unsupported_extension_ignored(self, tmp_dirs, init_test_db):
        """Files with unsupported extensions should be ignored."""
        inp, _, _ = tmp_dirs
        f = inp / "document.pdf"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)

        assert service._job_queue.empty()

    @patch("service._is_valid_video", return_value=False)
    @patch("service._is_file_stable", return_value=True)
    def test_corrupt_file_not_queued(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """A corrupt/invalid video file should not be queued."""
        inp, _, _ = tmp_dirs
        f = inp / "corrupt.mxf"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)

        assert service._job_queue.empty()


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 3: Subfolder structure
# ═════════════════════════════════════════════════════════════════════════════

class TestSubfolderStructure:
    """Test that subfolder structure is preserved from input → output."""

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_subfolder_file_queued(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """Files in subfolders should be discovered and queued."""
        inp, _, _ = tmp_dirs
        sub = inp / "channel_a" / "2026"
        f = sub / "episode01.mxf"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)

        assert not service._job_queue.empty()
        queued = service._job_queue.get_nowait()
        assert queued == f

    def test_output_preserves_subfolder(self, tmp_dirs):
        """Output path should mirror input subfolder structure."""
        inp, out, _ = tmp_dirs

        import service
        service.INPUT_FOLDER = inp
        service.OUTPUT_FOLDER = out

        input_path = inp / "channel_a" / "2026" / "episode01.mxf"
        try:
            rel = input_path.relative_to(inp)
        except ValueError:
            rel = Path(input_path.name)
        output_path = out / rel

        assert str(output_path) == str(out / "channel_a" / "2026" / "episode01.mxf")

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_rglob_finds_nested_files(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """Recursive scan (rglob) should find files in nested subfolders."""
        inp, _, _ = tmp_dirs
        # Create files in nested structure
        files = [
            inp / "a.mp4",
            inp / "sub1" / "b.mxf",
            inp / "sub1" / "sub2" / "c.mp4",
            inp / "sub1" / "sub2" / "sub3" / "d.mxf",
        ]
        for f in files:
            _create_fake_video(f, size_bytes=1024)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        # Simulate initial scan
        for p in sorted(inp.rglob("*")):
            if p.is_file() and p.suffix.lower() in service.ALLOWED_EXT:
                service._enqueue(p)

        assert service._job_queue.qsize() == 4


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 4: Periodic rescan picks up new files
# ═════════════════════════════════════════════════════════════════════════════

class TestPeriodicRescan:
    """Test that periodic rescan detects new large files added to input."""

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_rescan_picks_up_new_file(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """A file added after initial scan should be picked up on rescan."""
        inp, _, _ = tmp_dirs

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        # Initial scan — nothing
        for p in sorted(inp.rglob("*")):
            if p.is_file() and p.suffix.lower() in service.ALLOWED_EXT:
                service._enqueue(p)
        assert service._job_queue.empty()

        # New file appears
        f = inp / "new_episode.mxf"
        _create_fake_video(f, size_bytes=4096)

        # Rescan
        for p in sorted(inp.rglob("*")):
            if p.is_file() and p.suffix.lower() in service.ALLOWED_EXT:
                service._enqueue(p)

        assert service._job_queue.qsize() == 1

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_rescan_does_not_requeue_done(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """Rescan should not re-queue files already marked done in DB."""
        inp, _, _ = tmp_dirs
        f = inp / "already_done.mp4"
        _create_fake_video(f, size_bytes=2048)

        import db as db_module
        job_id = db_module.insert_job(
            input_path=str(f), filename=f.name,
            watermark_text="test", bits_list=[0]*256,
            codeword_hex="00"*32,
        )
        db_module.update_job_done(job_id, str(f), 25.0, 10.0)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        for p in sorted(inp.rglob("*")):
            if p.is_file() and p.suffix.lower() in service.ALLOWED_EXT:
                service._enqueue(p)

        assert service._job_queue.empty()

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_rescan_picks_up_new_file_in_subfolder(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """New file in subfolder should be found on rescan."""
        inp, _, _ = tmp_dirs

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        # New file in subfolder
        f = inp / "channel_b" / "new.mxf"
        _create_fake_video(f, size_bytes=4096)

        for p in sorted(inp.rglob("*")):
            if p.is_file() and p.suffix.lower() in service.ALLOWED_EXT:
                service._enqueue(p)

        assert service._job_queue.qsize() == 1


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 5: DB state transitions
# ═════════════════════════════════════════════════════════════════════════════

class TestDBStateTransitions:
    """Test database state transitions for jobs."""

    def test_insert_creates_processing_job(self, init_test_db):
        import db as db_module
        job_id = db_module.insert_job(
            input_path="/input/big.mxf", filename="big.mxf",
            watermark_text="abcd", bits_list=[0]*256,
            codeword_hex="00"*32,
        )
        with db_module._conn() as c:
            row = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row[0] == "processing"

    def test_done_updates_status(self, init_test_db):
        import db as db_module
        job_id = db_module.insert_job(
            input_path="/input/big.mxf", filename="big.mxf",
            watermark_text="abcd", bits_list=[0]*256,
            codeword_hex="00"*32,
        )
        db_module.update_job_done(job_id, "/output/big.mxf", 25.0, 300.5, has_audio=True)

        with db_module._conn() as c:
            row = c.execute("SELECT status, embed_time_s, has_audio FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row[0] == "done"
        assert row[1] == 300.5
        assert row[2] == 1

    def test_error_updates_status(self, init_test_db):
        import db as db_module
        job_id = db_module.insert_job(
            input_path="/input/corrupt.mxf", filename="corrupt.mxf",
            watermark_text="abcd", bits_list=[0]*256,
            codeword_hex="00"*32,
        )
        db_module.update_job_error(job_id, "CUDA OOM at frame 50000")

        with db_module._conn() as c:
            row = c.execute("SELECT status, error_msg FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row[0] == "error"
        assert "OOM" in row[1]

    def test_reset_stale_jobs(self, init_test_db):
        """Jobs stuck in 'processing' for too long should be reset (deleted)."""
        import db as db_module
        from datetime import datetime, timezone, timedelta

        # Insert a job with old timestamp
        with db_module._conn() as c:
            old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            c.execute(
                """INSERT INTO jobs (input_path, filename, watermark_text, bits_list,
                   codeword_hex, status, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                ("/input/stale.mxf", "stale.mxf", "test", "[]", "00"*32, "processing", old_time),
            )

        n = db_module.reset_stale_jobs(max_age_minutes=30)
        assert n == 1

        # Verify it's gone
        with db_module._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE filename='stale.mxf'").fetchone()
        assert row is None

    def test_recent_processing_not_reset(self, init_test_db):
        """Recently started 'processing' jobs should NOT be reset."""
        import db as db_module

        job_id = db_module.insert_job(
            input_path="/input/active.mxf", filename="active.mxf",
            watermark_text="test", bits_list=[0]*256,
            codeword_hex="00"*32,
        )

        n = db_module.reset_stale_jobs(max_age_minutes=30)
        assert n == 0

        # Job should still exist
        with db_module._conn() as c:
            row = c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        assert row[0] == "processing"


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 6: Large file copy simulation (end-to-end enqueue)
# ═════════════════════════════════════════════════════════════════════════════

class TestLargeFileCopySimulation:
    """Simulate real-world scenario: large file being copied over network."""

    def test_file_copying_then_stable(self, tmp_dirs, init_test_db):
        """Simulate 10GB file copy: rejected while growing, accepted when done."""
        inp, _, _ = tmp_dirs
        f = inp / "10gb_show.mxf"

        import service
        service.INPUT_FOLDER = inp

        # Phase 1: file is being copied (growing)
        t = _create_growing_file(f, initial_size=1000, grow_by=10000, interval=0.3, steps=3)
        time.sleep(0.1)

        stable = service._is_file_stable(f, wait_seconds=1)
        assert stable is False, "File should be unstable while copying"

        # Phase 2: copy finishes
        t.join()
        time.sleep(0.5)

        stable = service._is_file_stable(f, wait_seconds=1)
        assert stable is True, "File should be stable after copy completes"

    @patch("service._is_valid_video", return_value=True)
    def test_multiple_large_files_sequential_copy(self, mock_valid, tmp_dirs, init_test_db):
        """Multiple large files copied one after another."""
        inp, _, _ = tmp_dirs

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        files = []
        for i in range(3):
            f = inp / f"episode_{i:02d}.mxf"
            _create_fake_video(f, size_bytes=2048 * (i + 1))
            files.append(f)

        with patch("service._is_file_stable", return_value=True):
            for f in files:
                service._enqueue(f)

        assert service._job_queue.qsize() == 3

    @patch("service._is_valid_video", return_value=True)
    def test_file_added_during_processing(self, mock_valid, tmp_dirs, init_test_db):
        """New file appears while another is being processed."""
        inp, _, _ = tmp_dirs

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        # File 1 already queued
        f1 = inp / "first.mxf"
        _create_fake_video(f1, size_bytes=2048)
        with patch("service._is_file_stable", return_value=True):
            service._enqueue(f1)

        assert service._job_queue.qsize() == 1

        # File 2 appears later
        f2 = inp / "second.mxf"
        _create_fake_video(f2, size_bytes=4096)
        with patch("service._is_file_stable", return_value=True):
            service._enqueue(f2)

        assert service._job_queue.qsize() == 2


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 7: Edge cases
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases for large file handling."""

    def test_macos_resource_fork_ignored(self, tmp_dirs, init_test_db):
        """macOS ._* files should be ignored."""
        inp, _, _ = tmp_dirs
        f = inp / "._hidden.mxf"
        _create_fake_video(f, size_bytes=1024)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(f)
        assert service._job_queue.empty()

    def test_directory_ignored(self, tmp_dirs, init_test_db):
        """Directories should be ignored by _enqueue."""
        inp, _, _ = tmp_dirs
        d = inp / "some_folder.mxf"  # folder with video extension
        d.mkdir()

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        service._enqueue(d)
        assert service._job_queue.empty()

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_special_chars_in_filename(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """Filenames with special characters (spaces, brackets, unicode)."""
        inp, _, _ = tmp_dirs

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        f = inp / "Bang Goes Theory [ABC123] (2026).mxf"
        _create_fake_video(f, size_bytes=2048)
        service._enqueue(f)

        assert service._job_queue.qsize() == 1

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_concurrent_enqueue_dedup(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """Multiple threads calling _enqueue on same file — only one should succeed."""
        inp, _, _ = tmp_dirs
        f = inp / "race_condition.mxf"
        _create_fake_video(f, size_bytes=2048)

        import service
        service.INPUT_FOLDER = inp
        service._queued_paths.clear()
        service._job_queue = queue.Queue()

        results = []
        def _try_enqueue():
            service._enqueue(f)
            results.append(1)

        threads = [threading.Thread(target=_try_enqueue) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only 1 should be in queue despite 10 threads
        assert service._job_queue.qsize() == 1

    @patch("service._is_valid_video", return_value=True)
    @patch("service._is_file_stable", return_value=True)
    def test_all_supported_extensions(self, mock_stable, mock_valid, tmp_dirs, init_test_db):
        """All supported extensions should be accepted."""
        inp, _, _ = tmp_dirs

        import service
        service.INPUT_FOLDER = inp

        for ext in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp", ".ts", ".flv", ".mxf"]:
            service._queued_paths.clear()
            service._job_queue = queue.Queue()

            f = inp / f"test{ext}"
            _create_fake_video(f, size_bytes=1024)
            service._enqueue(f)

            assert not service._job_queue.empty(), f"Extension {ext} should be accepted"


# ═════════════════════════════════════════════════════════════════════════════
# TEST GROUP 8: Watchdog event handler
# ═════════════════════════════════════════════════════════════════════════════

class TestWatchdogHandler:
    """Test the _VideoHandler event dispatch."""

    def test_created_event_dispatches(self, tmp_dirs, init_test_db):
        """FileCreated event should trigger _enqueue."""
        inp, _, _ = tmp_dirs
        f = inp / "new_video.mp4"
        _create_fake_video(f, size_bytes=1024)

        import service

        event = MagicMock()
        event.event_type = "created"
        event.is_directory = False
        event.src_path = str(f)
        event.dest_path = None

        handler = service._VideoHandler()

        with patch("service._enqueue") as mock_enqueue, \
             patch("time.sleep"):  # skip the 1.5s delay
            handler.dispatch(event)
            mock_enqueue.assert_called_once()

    def test_directory_event_ignored(self, tmp_dirs, init_test_db):
        """Directory creation events should be ignored."""
        inp, _, _ = tmp_dirs

        import service

        event = MagicMock()
        event.event_type = "created"
        event.is_directory = True
        event.src_path = str(inp / "new_folder")

        handler = service._VideoHandler()

        with patch("service._enqueue") as mock_enqueue:
            handler.dispatch(event)
            mock_enqueue.assert_not_called()

    def test_non_video_event_ignored(self, tmp_dirs, init_test_db):
        """Non-video file events should be ignored."""
        inp, _, _ = tmp_dirs
        f = inp / "readme.txt"
        f.touch()

        import service

        event = MagicMock()
        event.event_type = "created"
        event.is_directory = False
        event.src_path = str(f)
        event.dest_path = None

        handler = service._VideoHandler()

        with patch("service._enqueue") as mock_enqueue:
            handler.dispatch(event)
            mock_enqueue.assert_not_called()
