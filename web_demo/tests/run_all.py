"""
Chạy tất cả test API.

Usage:
    cd web_demo/tests
    python run_all.py
"""

import subprocess
import sys

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

tests = [
    ("Video API", "test_video_api.py"),
    ("Audio API", "test_audio_api.py"),
]

results = []
for name, script in tests:
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  Chạy: {name}{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    ret = subprocess.call([sys.executable, script])
    results.append((name, ret == 0))

print(f"\n{BOLD}{'═'*55}{RESET}")
print(f"{BOLD}  Tổng kết{RESET}")
print(f"{BOLD}{'═'*55}{RESET}")
for name, passed in results:
    icon = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  {icon}  {name}")
print()

sys.exit(0 if all(p for _, p in results) else 1)
