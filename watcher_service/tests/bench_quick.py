"""
Quick benchmark: measure time breakdown of current pipeline.
Usage: python bench_quick.py --input ../data/val/gi-diploma-66013582898105388892353.mp4
"""
import argparse, logging, os, sys, time, tempfile
from pathlib import Path

import av, torch, numpy as np
from torch.nn import functional as F

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parent
for p in [str(_ROOT), str(_ROOT / "audioseal" / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["NO_TORCH_COMPILE"] = "1"

from videoseal.utils.cfg import setup_model_from_checkpoint
from web_demo.core.ecc import text_to_msg_tensor_bch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INTERP = dict(mode="bilinear", align_corners=False, antialias=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--chunk", type=int, default=45)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    # Load model
    ckpt = str(_ROOT / "output" / "run2_video" / "checkpoint350.pth")
    model = setup_model_from_checkpoint(ckpt)
    model.eval().to(DEVICE)
    msg_t, _, _ = text_to_msg_tensor_bch("bn0", msg_bits=256)
    msg_gpu = msg_t.to(DEVICE)
    msgs = msg_gpu[:1].repeat(model.chunk_size, 1)

    img_size = model.img_size
    step_size = model.step_size
    ck_size = model.chunk_size

    log.info(f"Model: img_size={img_size}, step_size={step_size}, chunk_size={ck_size}")
    log.info(f"VRAM after load: {torch.cuda.memory_allocated(0)/1024**2:.0f} MB")

    # Decode all frames first
    log.info(f"\nDecoding {args.input}...")
    frames_raw = []
    with av.open(args.input) as c:
        s = c.streams.video[0]
        s.thread_type = "FRAME"
        fps = float(s.average_rate or s.guessed_rate or 25)
        for f in c.decode(s):
            frames_raw.append(torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1))
    
    total = len(frames_raw)
    H, W = frames_raw[0].shape[1], frames_raw[0].shape[2]
    log.info(f"Decoded {total} frames, {W}x{H}, fps={fps:.1f}")

    chunk_size = args.chunk

    # ── Benchmark each stage ─────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"Stage-by-stage benchmark (chunk={chunk_size})")
    log.info(f"{'='*60}")

    # Process in chunks
    t_stack = 0; t_resize_down = 0; t_gpu_infer = 0
    t_resize_up = 0; t_blend = 0; t_to_numpy = 0; t_encode = 0

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out_stream = None

    with av.open(tmp, mode="w") as out:
        for ci in range(0, total, chunk_size):
            chunk_frames = frames_raw[ci:ci+chunk_size]
            n = len(chunk_frames)

            # Stack + normalize
            t0 = time.perf_counter()
            batch = torch.stack(chunk_frames).float() / 255.0
            t_stack += time.perf_counter() - t0

            N, C, Hb, Wb = batch.shape

            # Resize down
            t0 = time.perf_counter()
            if Hb != img_size or Wb != img_size:
                batch_256 = F.interpolate(batch, size=(img_size, img_size), **_INTERP)
            else:
                batch_256 = batch
            t_resize_down += time.perf_counter() - t0

            # GPU inference
            t0 = time.perf_counter()
            all_deltas = []
            for ii in range(0, len(batch_256[::step_size]), ck_size):
                nimgs = min(ck_size, len(batch_256[::step_size]) - ii)
                start = ii * step_size
                end = start + nimgs * step_size
                ck = batch_256[start:end].to(DEVICE)
                key = ck[::step_size]
                if model.embedder.yuv:
                    key = model.rgb2yuv(key)[:, 0:1]
                ck_msgs = msgs[:nimgs] if nimgs < ck_size else msgs
                with torch.no_grad():
                    preds = model.embedder(key, ck_msgs)
                preds = model._apply_video_mode(preds, len(ck), step_size, model.video_mode)
                if model.attenuation is not None:
                    model.attenuation.to(DEVICE)
                    hmaps = model.attenuation.heatmaps(ck)
                    preds = hmaps * preds
                all_deltas.append(preds.cpu())
                del ck, preds
            delta_256 = torch.cat(all_deltas, dim=0)[:N]
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t_gpu_infer += time.perf_counter() - t0

            # Resize up (CPU)
            t0 = time.perf_counter()
            if Hb != img_size or Wb != img_size:
                delta_full = F.interpolate(delta_256, size=(Hb, Wb), **_INTERP)
            else:
                delta_full = delta_256
            del delta_256
            t_resize_up += time.perf_counter() - t0

            # Blend (CPU)
            t0 = time.perf_counter()
            si = float(model.blender.scaling_i)
            sw = float(model.blender.scaling_w)
            result = si * batch + sw * delta_full
            if model.clamp:
                result = torch.clamp(result, 0, 1)
            del batch, delta_full
            t_blend += time.perf_counter() - t0

            # To numpy
            t0 = time.perf_counter()
            vid_np = (result * 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
            del result
            t_to_numpy += time.perf_counter() - t0

            # Encode
            t0 = time.perf_counter()
            if out_stream is None:
                h2 = vid_np.shape[1] - vid_np.shape[1] % 2
                w2 = vid_np.shape[2] - vid_np.shape[2] % 2
                out_stream = out.add_stream("h264", rate=int(fps))
                out_stream.width = w2
                out_stream.height = h2
                out_stream.pix_fmt = "yuv420p"
                out_stream.options = {"crf": "18", "preset": "fast"}
            for fnp in vid_np[:, :h2, :w2, :]:
                fr = av.VideoFrame.from_ndarray(fnp, format="rgb24")
                for pkt in out_stream.encode(fr):
                    out.mux(pkt)
            t_encode += time.perf_counter() - t0

        if out_stream:
            for pkt in out_stream.encode():
                out.mux(pkt)

    os.unlink(tmp)

    t_total = t_stack + t_resize_down + t_gpu_infer + t_resize_up + t_blend + t_to_numpy + t_encode

    log.info(f"\nResults ({total} frames, {W}x{H}):")
    log.info(f"  Stack+norm:   {t_stack:.2f}s  ({t_stack/t_total*100:5.1f}%)")
    log.info(f"  Resize down:  {t_resize_down:.2f}s  ({t_resize_down/t_total*100:5.1f}%)")
    log.info(f"  GPU infer:    {t_gpu_infer:.2f}s  ({t_gpu_infer/t_total*100:5.1f}%)")
    log.info(f"  Resize up:    {t_resize_up:.2f}s  ({t_resize_up/t_total*100:5.1f}%)")
    log.info(f"  Blend (CPU):  {t_blend:.2f}s  ({t_blend/t_total*100:5.1f}%)")
    log.info(f"  To numpy:     {t_to_numpy:.2f}s  ({t_to_numpy/t_total*100:5.1f}%)")
    log.info(f"  H264 encode:  {t_encode:.2f}s  ({t_encode/t_total*100:5.1f}%)")
    log.info(f"  ─────────────────────────────")
    log.info(f"  TOTAL:        {t_total:.2f}s  ({total/t_total:.1f} fps)")
    log.info(f"  GPU peak:     {torch.cuda.max_memory_allocated(0)/1024**2:.0f} MB")

    # ── Now test GPU blend ───────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"GPU Blend + FP16 test")
    log.info(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()
    t_gpu_total = 0

    fd2, tmp2 = tempfile.mkstemp(suffix=".mp4")
    os.close(fd2)
    out_stream2 = None

    with av.open(tmp2, mode="w") as out2:
        for ci in range(0, total, chunk_size):
            chunk_frames = frames_raw[ci:ci+chunk_size]
            n = len(chunk_frames)
            batch = torch.stack(chunk_frames).float() / 255.0
            N, C, Hb, Wb = batch.shape

            t0 = time.perf_counter()

            # Resize down on CPU
            if Hb != img_size or Wb != img_size:
                batch_256 = F.interpolate(batch, size=(img_size, img_size), **_INTERP)
            else:
                batch_256 = batch

            # GPU: inference with FP16
            all_deltas = []
            for ii in range(0, len(batch_256[::step_size]), ck_size):
                nimgs = min(ck_size, len(batch_256[::step_size]) - ii)
                start = ii * step_size
                end = start + nimgs * step_size
                ck = batch_256[start:end].to(DEVICE)
                key = ck[::step_size]
                if model.embedder.yuv:
                    key = model.rgb2yuv(key)[:, 0:1]
                ck_msgs = msgs[:nimgs] if nimgs < ck_size else msgs
                with torch.no_grad(), torch.cuda.amp.autocast():
                    preds = model.embedder(key, ck_msgs)
                preds = preds.float()  # back to fp32 for video_mode
                preds = model._apply_video_mode(preds, len(ck), step_size, model.video_mode)
                if model.attenuation is not None:
                    model.attenuation.to(DEVICE)
                    hmaps = model.attenuation.heatmaps(ck)
                    preds = hmaps * preds
                all_deltas.append(preds)  # keep on GPU!
                del ck
            delta_256_gpu = torch.cat(all_deltas, dim=0)[:N]

            # GPU: resize up + blend
            batch_gpu = batch.to(DEVICE, non_blocking=True)
            if Hb != img_size or Wb != img_size:
                delta_full_gpu = F.interpolate(delta_256_gpu, size=(Hb, Wb), **_INTERP)
            else:
                delta_full_gpu = delta_256_gpu
            del delta_256_gpu

            si = float(model.blender.scaling_i)
            sw = float(model.blender.scaling_w)
            result_gpu = si * batch_gpu + sw * delta_full_gpu
            if model.clamp:
                result_gpu = torch.clamp(result_gpu, 0, 1)
            del batch_gpu, delta_full_gpu

            # GPU→CPU
            vid_np = (result_gpu * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            del result_gpu

            torch.cuda.synchronize()
            t_gpu_total += time.perf_counter() - t0

            # Encode
            if out_stream2 is None:
                h2 = vid_np.shape[1] - vid_np.shape[1] % 2
                w2 = vid_np.shape[2] - vid_np.shape[2] % 2
                out_stream2 = out2.add_stream("h264", rate=int(fps))
                out_stream2.width = w2
                out_stream2.height = h2
                out_stream2.pix_fmt = "yuv420p"
                out_stream2.options = {"crf": "18", "preset": "fast"}
            for fnp in vid_np[:, :h2, :w2, :]:
                fr = av.VideoFrame.from_ndarray(fnp, format="rgb24")
                for pkt in out_stream2.encode(fr):
                    out2.mux(pkt)

        if out_stream2:
            for pkt in out_stream2.encode():
                out2.mux(pkt)

    os.unlink(tmp2)

    log.info(f"  GPU path (infer+resize+blend): {t_gpu_total:.2f}s ({total/t_gpu_total:.1f} fps)")
    log.info(f"  GPU peak: {torch.cuda.max_memory_allocated(0)/1024**2:.0f} MB")


if __name__ == "__main__":
    main()
