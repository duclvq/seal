"""Quick verify: embed → extract with official extract_message()"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, 'audioseal/src')
sys.path.insert(0, 'watcher_service')
os.environ['NO_TORCH_COMPILE'] = '1'

import torch
from worker import load_video_model, embed_to_file
from web_demo.core.ecc import msg_tensor_to_text_bch

device = torch.device('cuda')
model = load_video_model('output/run2_video/checkpoint350.pth', device)

fd, out = tempfile.mkstemp(suffix='.mp4')
os.close(fd)
r = embed_to_file(
    r'D:\samu_videos_gpu\DJI_20260104092139_0153_D_gpu_hevc.mp4',
    out, model, 'test', device, None
)
print(f"Embed: {r['mode']}")

# Streaming extract with running aggregation (low memory)
import av
running_sum = None
n_frames = 0
CHUNK = 8
with av.open(out) as container:
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    chunk_buf = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(rgb).permute(2, 0, 1)  # uint8
        chunk_buf.append(t)
        if len(chunk_buf) >= CHUNK:
            video_chunk = torch.stack(chunk_buf).to(device).float().div_(255.0)
            chunk_buf.clear()
            with torch.no_grad():
                outputs = model.detect(video_chunk, is_video=True)
            preds = outputs["preds"]
            if preds.dim() == 4:
                preds = preds.mean(dim=(-2, -1))
            bp = preds[:, 1:].cpu()
            cs = (bp * bp.abs()).sum(dim=0)
            running_sum = cs if running_sum is None else running_sum + cs
            n_frames += bp.shape[0]
            del video_chunk, outputs, preds, bp, cs
            torch.cuda.empty_cache()
        if n_frames >= 300:
            break
    if chunk_buf:
        video_chunk = torch.stack(chunk_buf).to(device).float().div_(255.0)
        chunk_buf.clear()
        with torch.no_grad():
            outputs = model.detect(video_chunk, is_video=True)
        preds = outputs["preds"]
        if preds.dim() == 4:
            preds = preds.mean(dim=(-2, -1))
        bp = preds[:, 1:].cpu()
        cs = (bp * bp.abs()).sum(dim=0)
        running_sum = cs if running_sum is None else running_sum + cs
        n_frames += bp.shape[0]
        del video_chunk, outputs, preds, bp, cs

print(f"Extracted {n_frames} frames (streaming)")
msg = (running_sum / n_frames > 0).unsqueeze(0)
decode = msg_tensor_to_text_bch(msg)
print(f"Detected: {decode['correctable']}")
print(f"Text: '{decode['decoded_text']}'")
print(f"Bits: {decode.get('bits_list', [])[:20]}")

os.unlink(out)
