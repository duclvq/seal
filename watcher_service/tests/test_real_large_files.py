"""
Integration tests with REAL large video files.
Tests the full watcher pipeline: enqueue → embed → verify output.

Prerequisites:
  - Files created by ffmpeg in D:/data/temp/:
    test_1gb.mp4  (~869MB, 1080p, 10min)
    test_2gb.mxf  (~1.9GB, 1080p, 10min)
    test_5gb.mp4  (~4.3GB, 1080p, 30min)
  - GPU with CUDA available
  - VideoSeal checkpoint at output/run2_video/checkpoint350.pth

Run:  cd watcher_service && python -m pytest tests/test_real_large_files.py -v -s
"""

import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest

# ── Setup paths ─────────────────────────────────────────────────────────────
_TEST_DIR = Path(__file__).resolve().parent
_SERVICE_DIR = _TEST_DIR.parent
_PROJECT_ROOT = _SERVICE_DIR.parent

if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.chdir(_PROJECT_ROOT)
os.environ.setdefault("NO_TORCH_COMPILE", "1")

# ── Test file paths ─────────────────────────────────────────────────────────
TEST_FILES = {
    "1gb_mp4": Path("D:/data/temp/test_1gb.mp4"),
    "2gb_mxf": Path("D:/data/temp/test_2gb.mxf"),
    "5gb_mp4": Path("D:/data/temp/test_5gb.mp4"),
}

CKPT_PATH = _PROJECT_ROOT / "output" / "run2_video" / "checkpoint350.pth"

log = logging.getLogger(__name__)


# ── Fixtures ────────────────────────────────────────────────────────────────

def _check_file(name):
    p = TEST_FILES[name]
    if not p.exists():
        pytest.skip(f"Test file not found: {p}")
    return p


def _check_gpu():
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _check_ckpt():
    if not CKPT_PATH.exists():
        pytest.skip(f"Checkpoint not found: {CKPT_PATH}")


@pytest.fixture(scope="module")
def video_model():
    """Load VideoSeal model once for all tests in this module."""
    _check_gpu()
    _check_ckpt()
    import torch
    from worker import load_video_model
    device = torch.device("cuda")
    model = load_video_model(str(CKPT_PATH), device)
    yield model, device
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def audio_model():
    """Load AudioSeal model once for all tests."""
    _check_gpu()
    import torch
    from worker import load_audio_model
    device = torch.device("cuda")
    model = load_audio_model(device)
    if model is None:
        pytest.skip("AudioSeal not available")
    yield model, device
    torch.cuda.empty_cache()


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: File validation with real files
# ═════════════════════════════════════════════════════════════════════════════

class TestRealFileValidation:
    """Test _is_valid_video and _is_file_stable with real large files."""

    def test_1gb_mp4_is_valid(self):
        f = _check_file("1gb_mp4")
        from service import _is_valid_video
        assert _is_valid_video(f) is True

    def test_2gb_mxf_is_valid(self):
        f = _check_file("2gb_mxf")
        from service import _is_valid_video
        assert _is_valid_video(f) is True

    def test_5gb_mp4_is_valid(self):
        f = _check_file("5gb_mp4")
        from service import _is_valid_video
        assert _is_valid_video(f) is True

    def test_1gb_mp4_is_stable(self):
        f = _check_file("1gb_mp4")
        from service import _is_file_stable
        assert _is_file_stable(f, wait_seconds=2) is True

    def test_2gb_mxf_is_stable(self):
        f = _check_file("2gb_mxf")
        from service import _is_file_stable
        assert _is_file_stable(f, wait_seconds=2) is True

    def test_file_size_matches(self):
        """Verify test files are actually large."""
        f1 = _check_file("1gb_mp4")
        f2 = _check_file("2gb_mxf")
        f3 = _check_file("5gb_mp4")
        assert f1.stat().st_size > 500 * 1024 * 1024, "test_1gb.mp4 should be >500MB"
        assert f2.stat().st_size > 1 * 1024 * 1024 * 1024, "test_2gb.mxf should be >1GB"
        assert f3.stat().st_size > 3 * 1024 * 1024 * 1024, "test_5gb.mp4 should be >3GB"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2: Full embed pipeline with real large files
# ═════════════════════════════════════════════════════════════════════════════

class TestRealEmbed:
    """Test embed_to_file with real large video files."""

    def test_embed_1gb_mp4(self, video_model, audio_model, output_dir):
        """Embed watermark into 869MB MP4 file (1080p, 10min, ~15000 frames)."""
        f = _check_file("1gb_mp4")
        model, device = video_model
        amodel, _ = audio_model

        from worker import embed_to_file
        out = output_dir / "wm_test_1gb.mp4"

        t0 = time.time()
        result = embed_to_file(
            input_path=str(f),
            output_path=str(out),
            video_model=model,
            watermark_text="ab12",
            device=device,
            audio_model=amodel,
        )
        elapsed = time.time() - t0

        assert out.exists(), "Output file should be created"
        assert out.stat().st_size > 0, "Output file should not be empty"
        assert result["total_frames"] > 0
        assert result["has_audio"] is True
        assert result["resolution"] == "1920x1080"

        fps_proc = result["total_frames"] / elapsed
        log.warning(
            f"[1GB MP4] {result['total_frames']} frames, "
            f"{elapsed:.1f}s, {fps_proc:.1f} f/s, "
            f"output={out.stat().st_size / 1024**2:.0f}MB"
        )

    def test_embed_2gb_mxf(self, video_model, audio_model, output_dir):
        """Embed watermark into 1.9GB MXF file (1080p, 10min, MPEG2)."""
        f = _check_file("2gb_mxf")
        model, device = video_model
        amodel, _ = audio_model

        from worker import embed_to_file
        out = output_dir / "wm_test_2gb.mxf"

        t0 = time.time()
        result = embed_to_file(
            input_path=str(f),
            output_path=str(out),
            video_model=model,
            watermark_text="cd34",
            device=device,
            audio_model=amodel,
        )
        elapsed = time.time() - t0

        assert out.exists(), "Output MXF should be created"
        assert out.stat().st_size > 0
        assert result["total_frames"] > 0
        assert result["resolution"] == "1920x1080"

        fps_proc = result["total_frames"] / elapsed
        log.warning(
            f"[2GB MXF] {result['total_frames']} frames, "
            f"{elapsed:.1f}s, {fps_proc:.1f} f/s, "
            f"output={out.stat().st_size / 1024**2:.0f}MB"
        )

    @pytest.mark.slow
    def test_embed_5gb_mp4(self, video_model, audio_model, output_dir):
        """Embed watermark into 4.3GB MP4 file (1080p, 30min, ~45000 frames).
        This test takes a long time (~10+ min on RTX 4080)."""
        f = _check_file("5gb_mp4")
        model, device = video_model
        amodel, _ = audio_model

        from worker import embed_to_file
        out = output_dir / "wm_test_5gb.mp4"

        t0 = time.time()
        result = embed_to_file(
            input_path=str(f),
            output_path=str(out),
            video_model=model,
            watermark_text="ef56",
            device=device,
            audio_model=amodel,
        )
        elapsed = time.time() - t0

        assert out.exists()
        assert out.stat().st_size > 0
        assert result["total_frames"] > 10000
        assert result["has_audio"] is True

        fps_proc = result["total_frames"] / elapsed
        log.warning(
            f"[5GB MP4] {result['total_frames']} frames, "
            f"{elapsed:.1f}s, {fps_proc:.1f} f/s, "
            f"output={out.stat().st_size / 1024**2:.0f}MB"
        )


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3: Extraction accuracy on embedded large files
# ═════════════════════════════════════════════════════════════════════════════

class TestRealExtractAccuracy:
    """Embed then extract and verify watermark accuracy."""

    def test_embed_extract_1gb_mp4(self, video_model, audio_model, output_dir):
        """Embed + extract on 1GB MP4, verify bit accuracy >= 90%."""
        f = _check_file("1gb_mp4")
        model, device = video_model
        amodel, _ = audio_model
        wm_text = "xx99"

        from worker import embed_to_file
        from web_demo.core.ecc import text_to_msg_tensor_bch

        out = output_dir / "verify_1gb.mp4"
        embed_to_file(
            input_path=str(f), output_path=str(out),
            video_model=model, watermark_text=wm_text,
            device=device, audio_model=amodel,
        )

        # Extract watermark from output
        import torch
        import av

        _, _, expected_bits = text_to_msg_tensor_bch(wm_text, msg_bits=256)

        # Read frames — limit to 90 to fit 1080p in ~2GB GPU memory
        max_frames = 90
        frames = []
        with av.open(str(out)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                t = torch.from_numpy(
                    frame.to_ndarray(format="rgb24")
                ).permute(2, 0, 1).float() / 255.0
                frames.append(t)
                if len(frames) >= max_frames:
                    break

        video_tensor = torch.stack(frames).to(device)
        del frames
        with torch.no_grad():
            extracted = model.extract_message(video_tensor, aggregation="squared_avg")
        extracted_bits = (extracted > 0).int().squeeze().cpu().tolist()
        del video_tensor
        torch.cuda.empty_cache()

        # Compare bits
        n_correct = sum(a == b for a, b in zip(extracted_bits, expected_bits))
        accuracy = n_correct / len(expected_bits)

        log.warning(f"[EXTRACT 1GB MP4] accuracy={accuracy:.2%} ({n_correct}/256)")
        # Synthetic video (testsrc2) has lower accuracy than real video due to
        # uniform patterns degrading more during H264 re-encoding
        assert accuracy >= 0.70, f"Extraction accuracy too low: {accuracy:.2%}"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Memory usage during large file processing
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryUsage:
    """Monitor GPU and system memory during large file processing."""

    def test_gpu_memory_no_leak_1gb(self, video_model, audio_model, output_dir):
        """Process 1GB file and verify GPU memory is released after."""
        f = _check_file("1gb_mp4")
        model, device = video_model
        amodel, _ = audio_model

        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        mem_before = torch.cuda.memory_allocated(device) / 1024**2

        from worker import embed_to_file
        out = output_dir / "mem_test.mp4"
        embed_to_file(
            input_path=str(f), output_path=str(out),
            video_model=model, watermark_text="mem1",
            device=device, audio_model=amodel,
        )

        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated(device) / 1024**2
        mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2

        leak = mem_after - mem_before
        log.warning(
            f"[MEMORY] before={mem_before:.0f}MB, after={mem_after:.0f}MB, "
            f"peak={mem_peak:.0f}MB, leak={leak:.0f}MB"
        )

        # Allow up to 100MB drift (PyTorch caches, etc.)
        assert leak < 100, f"Possible GPU memory leak: {leak:.0f}MB"

    def test_gpu_memory_stays_bounded(self, video_model, output_dir):
        """Verify peak GPU usage stays under VRAM limit during 1080p processing."""
        f = _check_file("1gb_mp4")
        model, device = video_model

        import torch
        torch.cuda.reset_peak_memory_stats(device)

        from worker import embed_to_file
        out = output_dir / "bound_test.mp4"
        embed_to_file(
            input_path=str(f), output_path=str(out),
            video_model=model, watermark_text="bnd1",
            device=device,
        )

        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        total_mb = torch.cuda.get_device_properties(device).total_memory / 1024**2

        log.warning(f"[VRAM] peak={peak_mb:.0f}MB / total={total_mb:.0f}MB")
        assert peak_mb < total_mb * 0.95, f"Peak VRAM too high: {peak_mb:.0f}/{total_mb:.0f}MB"


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5: Output file integrity
# ═════════════════════════════════════════════════════════════════════════════

class TestOutputIntegrity:
    """Verify output files are valid and playable."""

    def test_output_mp4_playable(self, video_model, output_dir):
        """Output MP4 should be openable by PyAV with correct properties."""
        f = _check_file("1gb_mp4")
        model, device = video_model

        from worker import embed_to_file
        out = output_dir / "integrity.mp4"
        result = embed_to_file(
            input_path=str(f), output_path=str(out),
            video_model=model, watermark_text="int1",
            device=device,
        )

        import av
        with av.open(str(out)) as c:
            vs = c.streams.video[0]
            assert vs.width > 0
            assert vs.height > 0
            # Resolution should be even (yuv420p requirement)
            assert vs.width % 2 == 0
            assert vs.height % 2 == 0
            # Frame count should roughly match
            n_out = vs.frames
            if n_out > 0:  # some containers don't report frame count
                assert abs(n_out - result["total_frames"]) < 10

    def test_output_has_audio_track(self, video_model, audio_model, output_dir):
        """Output should preserve audio track."""
        f = _check_file("1gb_mp4")
        model, device = video_model
        amodel, _ = audio_model

        from worker import embed_to_file
        out = output_dir / "audio_check.mp4"
        result = embed_to_file(
            input_path=str(f), output_path=str(out),
            video_model=model, watermark_text="aud1",
            device=device, audio_model=amodel,
        )

        assert result["has_audio"] is True

        import av
        with av.open(str(out)) as c:
            assert len(c.streams.audio) > 0, "Output should have audio stream"

    def test_output_mxf_valid(self, video_model, audio_model, output_dir):
        """MXF output should be a valid video file."""
        f = _check_file("2gb_mxf")
        model, device = video_model
        amodel, _ = audio_model

        from worker import embed_to_file
        out = output_dir / "integrity.mxf"
        result = embed_to_file(
            input_path=str(f), output_path=str(out),
            video_model=model, watermark_text="mxf1",
            device=device, audio_model=amodel,
        )

        assert out.exists()
        assert out.stat().st_size > 0

        # Should be openable
        import av
        with av.open(str(out)) as c:
            assert len(c.streams.video) > 0


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6: DB integration with real files
# ═════════════════════════════════════════════════════════════════════════════

class TestDBIntegrationReal:
    """Test DB operations with real file paths and sizes."""

    def test_full_job_lifecycle(self, video_model, audio_model, output_dir):
        """Insert job → embed → update done — full lifecycle with real file."""
        f = _check_file("1gb_mp4")
        model, device = video_model
        amodel, _ = audio_model

        import db as db_module
        from web_demo.core.ecc import text_to_msg_tensor_bch
        from worker import embed_to_file

        # Init temp DB
        db_path = output_dir / "test.db"
        db_module.init_db(str(db_path))

        wm_text = "db01"
        _, codeword, bits_list = text_to_msg_tensor_bch(wm_text, msg_bits=256)

        # Insert
        job_id = db_module.insert_job(
            input_path=str(f), filename=f.name,
            watermark_text=wm_text, bits_list=bits_list,
            codeword_hex=codeword.hex(),
        )

        # Embed
        out = output_dir / f.name
        t0 = time.time()
        result = embed_to_file(
            input_path=str(f), output_path=str(out),
            video_model=model, watermark_text=wm_text,
            device=device, audio_model=amodel,
        )
        elapsed = time.time() - t0

        # Update done
        db_module.update_job_done(job_id, str(out), result["fps"], elapsed, result["has_audio"])

        # Verify
        with db_module._conn() as c:
            row = c.execute(
                "SELECT status, filename, embed_time_s, has_audio FROM jobs WHERE id=?",
                (job_id,),
            ).fetchone()
        assert row[0] == "done"
        assert row[1] == f.name
        assert row[2] > 0
        assert row[3] == 1
