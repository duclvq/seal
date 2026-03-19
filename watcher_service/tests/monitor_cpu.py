"""Monitor CPU + RAM usage every 1 second for 35 seconds."""
import psutil, time
for _ in range(35):
    ts = time.strftime("%H:%M:%S")
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    total_gb = mem.total / 1024**3
    print(f"{ts} | CPU={cpu:5.1f}% | RAM={used_gb:.1f}/{total_gb:.0f}GB ({mem.percent}%)")
