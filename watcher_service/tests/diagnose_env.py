"""
Diagnostic script — chạy trên máy đối tác để kiểm tra môi trường.
Không cần import torch (tránh bị treo).

Cách dùng:
  1. Mở CMD
  2. cd D:\vtv_setup\seal_1603\seal_1603
  3. call base_conda\Scripts\activate.bat
  4. python watcher_service\diagnose_env.py
"""
import os
import sys
import subprocess
import ctypes
import platform

print("=" * 60)
print("ENVIRONMENT DIAGNOSTIC")
print("=" * 60)

# 1. Python info
print(f"\n[1] Python")
print(f"  Executable: {sys.executable}")
print(f"  Version:    {sys.version}")
print(f"  Platform:   {platform.platform()}")
print(f"  CWD:        {os.getcwd()}")

# 2. Conda env
print(f"\n[2] Conda Environment")
conda_prefix = os.environ.get("CONDA_PREFIX", "(not set)")
conda_env = os.environ.get("CONDA_DEFAULT_ENV", "(not set)")
print(f"  CONDA_PREFIX:      {conda_prefix}")
print(f"  CONDA_DEFAULT_ENV: {conda_env}")
print(f"  PATH (first 3):")
for i, p in enumerate(os.environ.get("PATH", "").split(os.pathsep)[:5]):
    print(f"    [{i}] {p}")

# 3. NVIDIA driver
print(f"\n[3] NVIDIA Driver")
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                                    "--format=csv,noheader"], timeout=10)
    for line in out.decode().strip().split("\n"):
        print(f"  {line.strip()}")
except FileNotFoundError:
    print("  nvidia-smi NOT FOUND — NVIDIA driver not installed?")
except Exception as e:
    print(f"  nvidia-smi error: {e}")

# 4. CUDA DLLs
print(f"\n[4] CUDA DLLs")
cuda_dlls = [
    "nvcuda.dll",       # CUDA driver
    "cudart64_12.dll",  # CUDA runtime 12.x
    "cudart64_11.dll",  # CUDA runtime 11.x
    "cublas64_12.dll",
    "cublas64_11.dll",
    "cudnn64_9.dll",    # cuDNN 9.x
    "cudnn64_8.dll",    # cuDNN 8.x
]
for dll in cuda_dlls:
    try:
        h = ctypes.WinDLL(dll)
        print(f"  {dll:30s} OK")
    except OSError:
        print(f"  {dll:30s} NOT FOUND")

# 5. PyTorch files check (without importing)
print(f"\n[5] PyTorch Installation")
torch_paths = []
for p in sys.path:
    torch_dir = os.path.join(p, "torch")
    if os.path.isdir(torch_dir):
        torch_paths.append(torch_dir)
if torch_paths:
    for tp in torch_paths:
        print(f"  Found: {tp}")
        # Check _C extension
        for ext in [".pyd", ".so"]:
            c_path = os.path.join(tp, f"_C{ext}")
            if os.path.exists(c_path):
                size_mb = os.path.getsize(c_path) / 1024 / 1024
                print(f"    _C{ext}: {size_mb:.1f}MB")
        # Check version
        ver_file = os.path.join(tp, "version.py")
        if os.path.exists(ver_file):
            with open(ver_file) as f:
                for line in f:
                    if line.startswith("__version__") or line.startswith("cuda"):
                        print(f"    {line.strip()}")
        # Check CUDA libs bundled with PyTorch
        lib_dir = os.path.join(tp, "lib")
        if os.path.isdir(lib_dir):
            cuda_libs = [f for f in os.listdir(lib_dir) if "cuda" in f.lower() or "cublas" in f.lower() or "cudnn" in f.lower()]
            if cuda_libs:
                print(f"    Bundled CUDA libs ({len(cuda_libs)}):")
                for cl in sorted(cuda_libs)[:10]:
                    print(f"      {cl}")
                if len(cuda_libs) > 10:
                    print(f"      ... and {len(cuda_libs)-10} more")
else:
    print("  PyTorch NOT FOUND in sys.path")

# 6. Try import torch with timeout
print(f"\n[6] Import torch test (10s timeout)")
print("  Starting...")
import threading
_result = {"ok": False, "version": "", "cuda": "", "error": ""}

def _try_import():
    try:
        import torch
        _result["ok"] = True
        _result["version"] = torch.__version__
        _result["cuda"] = str(torch.version.cuda)
        _result["cuda_avail"] = str(torch.cuda.is_available())
    except Exception as e:
        _result["error"] = str(e)

t = threading.Thread(target=_try_import, daemon=True)
t.start()
t.join(timeout=10)

if t.is_alive():
    print("  TIMEOUT — import torch bị treo sau 10s!")
    print("  → Nguyên nhân: CUDA driver không tương thích hoặc DLL bị thiếu/block")
    print("  → Thử: set CUDA_VISIBLE_DEVICES=-1 rồi chạy lại")
else:
    if _result["ok"]:
        print(f"  OK — PyTorch {_result['version']}, CUDA {_result['cuda']}, available={_result.get('cuda_avail','?')}")
    else:
        print(f"  FAILED — {_result['error']}")

# 7. Network check
print(f"\n[7] Network")
print(f"  HF_HUB_OFFLINE:      {os.environ.get('HF_HUB_OFFLINE', '(not set)')}")
print(f"  TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE', '(not set)')}")

# 8. Key files check
print(f"\n[8] Key Files")
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
key_files = [
    "watcher_service/service.py",
    "watcher_service/worker.py",
    "watcher_service/config.yaml",
    "watcher_service/.env",
    "output/run2_video/checkpoint350.pth",
]
for kf in key_files:
    fp = os.path.join(base, kf)
    if os.path.exists(fp):
        size = os.path.getsize(fp)
        if size > 1024*1024:
            print(f"  {kf:50s} {size/1024/1024:.1f}MB")
        else:
            print(f"  {kf:50s} {size/1024:.1f}KB")
    else:
        print(f"  {kf:50s} MISSING!")

print(f"\n{'=' * 60}")
print("Gửi output này cho dev team để phân tích.")
print("=" * 60)
