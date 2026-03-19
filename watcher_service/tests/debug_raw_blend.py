"""
Debug: simulate the GPU blend pipeline for chunk 1 and compare with actual output.
Identify exactly where the corruption happens (frames 20-31).
"""
import sys, os
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SERVICE_DIR.parent
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_AUDIOSEAL_SRC = _PROJECT_ROOT / "audioseal" / "src"
if str(_AUDIOSEAL_SRC) not in sys.path:
    sys.path.insert(0, str(_AUDIOSEAL_SRC))

os.environ.setdefault("NO_TORCH_COMPILE", "1")
os.chdir(_PROJECT_ROOT)

import av
import numpy as np
import torch
import torch.nn.functional as F
from worker import (load_video_model, _get_watermark_delta, _encode_audio_msg,
                    _yuv420p_to_rgb_gpu, _yuv420p_resize_down_gpu,
                    _BT709_FWD, _INTERP)

CKPT  = "output/run2_video/checkpoint350.pth"
INPUT = r"D:\samu_videos_gpu\DJI_20260104091947_0152_D_gpu_hevc.mp4"
OUTPUT = str(_SERVICE_DIR / "data" / "test_output" / "DJI_0152_wm.mp4")
device = torch.device("cuda")

print("Loading model...")
model = load_video_model(CKPT, device)
scaling_w = float(model.blender.scaling_w)
scaling_i = float(model.blender.scaling_i)
print(f"scaling_w={scaling_w}, scaling_i={scaling_i}")

# Extract first 32 frames as YUV420P
print("\nExtracting first 32 input frames (YUV420P)...")
frames_yuv = []
with av.open(INPUT) as c:
    s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i >= 32: break
        frames_yuv.append(f.to_ndarray(format="yuv420p"))

yuv_batch = torch.from_numpy(np.stack(frames_yuv))
N, total_h, W = yuv_batch.shape
H = (total_h * 2) // 3
print(f"YUV: {yuv_batch.shape}, H={H}, W={W}")

# Simulate GPU pipeline exactly like _gpu_worker
gpu_yuv = yuv_batch.to(device)
img_size = 256

# Step 2: resize down
yuv_256 = _yuv420p_resize_down_gpu(gpu_yuv, img_size, img_size)
# Step 3: yuv2rgb
rgb_256 = _yuv420p_to_rgb_gpu(yuv_256, device)
rgb_256_f = rgb_256.float().div_(255.0)
del rgb_256, yuv_256

# Step 4: embedder
msg_gpu = _encode_audio_msg("test", device)
delta_256 = _get_watermark_delta(rgb_256_f, model, msg_gpu, device, keep_on_gpu=True)
del rgb_256_f

# Step 5: delta -> YUV
if _BT709_FWD.device != device:
    _BT709_FWD = _BT709_FWD.to(device)

delta_256.mul_(scaling_w * 255.0)
dC = delta_256.shape[1]
print(f"\ndelta channels: {dC}")

if dC == 1:
    dY = delta_256
    dCb = torch.zeros_like(dY)
    dCr = torch.zeros_like(dY)
else:
    d_hwc = delta_256.permute(0, 2, 3, 1)
    d_yuv = d_hwc @ _BT709_FWD.T
    dY = d_yuv[..., 0].unsqueeze(1)
    dCb = d_yuv[..., 1].unsqueeze(1)
    dCr = d_yuv[..., 2].unsqueeze(1)
del delta_256

# Step 6: resize up
hh, hw = H // 2, W // 2
dY_full = F.interpolate(dY, size=(H, W), **_INTERP)
if dC > 1:
    dCb_full = F.interpolate(dCb, size=(hh, hw), **_INTERP)
    dCr_full = F.interpolate(dCr, size=(hh, hw), **_INTERP)
del dY, dCb, dCr

# Step 7: blend
Y_orig = gpu_yuv[:, :H, :].float()
Y_out = (scaling_i * Y_orig + dY_full.squeeze(1)).clamp_(0, 255).to(torch.uint8)
del Y_orig, dY_full

out_packed = gpu_yuv.clone()
out_packed[:, :H, :] = Y_out
del Y_out

if dC > 1:
    Cb_orig = gpu_yuv[:, H:, :hw].float()
    Cr_orig = gpu_yuv[:, H:, hw:].float()
    Cb_out = (scaling_i * Cb_orig + dCb_full.squeeze(1)).clamp_(0, 255).to(torch.uint8)
    Cr_out = (scaling_i * Cr_orig + dCr_full.squeeze(1)).clamp_(0, 255).to(torch.uint8)
    out_packed[:, H:, :hw] = Cb_out
    out_packed[:, H:, hw:] = Cr_out

# Download to CPU
result_cpu = out_packed.cpu().numpy()
del out_packed

# Now compare with actual output file
print("\nExtracting first 32 output frames...")
out_frames = []
with av.open(OUTPUT) as c:
    s = c.streams.video[0]
    for i, f in enumerate(c.decode(s)):
        if i >= 32: break
        out_frames.append(f.to_ndarray(format="yuv420p"))

print(f"Got {len(out_frames)} output frames")

# Compare: simulated blend vs actual output vs input
print("\n=== Comparison: Input vs Simulated vs Actual Output ===")
print("Frame  PSNR(in→sim)  PSNR(in→out)  PSNR(sim→out)  sim_Y_diff  out_Y_diff")
for i in range(min(32, len(out_frames))):
    inp_y = frames_yuv[i][:H, :].astype(np.float32)
    sim_y = result_cpu[i][:H, :].astype(np.float32)
    out_y = out_frames[i][:H, :].astype(np.float32)

    mse_in_sim = np.mean((inp_y - sim_y) ** 2)
    mse_in_out = np.mean((inp_y - out_y) ** 2)
    mse_sim_out = np.mean((sim_y - out_y) ** 2)

    psnr_in_sim = 10 * np.log10(255**2 / mse_in_sim) if mse_in_sim > 0 else 99
    psnr_in_out = 10 * np.log10(255**2 / mse_in_out) if mse_in_out > 0 else 99
    psnr_sim_out = 10 * np.log10(255**2 / mse_sim_out) if mse_sim_out > 0 else 99

    sim_diff = np.mean(np.abs(inp_y - sim_y))
    out_diff = np.mean(np.abs(inp_y - out_y))

    print(f"  {i:4d}  {psnr_in_sim:10.2f}  {psnr_in_out:10.2f}  {psnr_sim_out:11.2f}  "
          f"{sim_diff:10.2f}  {out_diff:10.2f}")

torch.cuda.empty_cache()
print("\nDone.")
