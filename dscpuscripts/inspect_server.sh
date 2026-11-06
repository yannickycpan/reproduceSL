#!/usr/bin/env bash
# Nonprivileged hardware, memory, Python, and disk diagnostic for direct servers.
set -euo pipefail
echo "=== Host and OS ==="
hostname
uname -srm
echo "=== CPU topology ==="
command -v lscpu >/dev/null 2>&1 && lscpu | grep -E '^(Architecture|Model name|Socket|Core|Thread|CPU\(s\)|NUMA node\(s\))' || true
echo "Available logical CPUs (affinity): $(nproc)"
echo "=== Memory ==="
command -v free >/dev/null 2>&1 && free -h || grep -E '^(MemTotal|MemAvailable|SwapTotal):' /proc/meminfo
echo "=== Disk ==="
df -h "$HOME"
echo "=== Python ==="
for cmd in python3.12 python3; do
  if command -v "$cmd" >/dev/null 2>&1; then
    "$cmd" --version
    "$cmd" -c 'import os; print("CPU affinity count:", len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())'
  fi
done
echo "=== Tools ==="
for cmd in git rsync tmux; do command -v "$cmd" || true; done
echo "=== Existing CPU environment ==="
if [[ -x "$HOME/scratch/fulltd/.venv-fulltd/bin/python" ]]; then
  "$HOME/scratch/fulltd/.venv-fulltd/bin/python" -c 'import torch; print("PyTorch:",torch.__version__)' || true
fi
