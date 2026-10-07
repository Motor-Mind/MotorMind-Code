#!/bin/bash
# Install what is not on PyPI into third_party/ (ignored by git), in the active env:
#   LIBERO at the commit the sweeps ran on (editable; its meshes are what the simulator loads).
#   --curobo: also cuRobo 0.8.0 for the xArm6 pre-flight (needs torch with CUDA).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd); TP=$ROOT/third_party; mkdir -p "$TP"
if [ ! -d "$TP/LIBERO/.git" ]; then git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$TP/LIBERO"; fi
git -C "$TP/LIBERO" checkout -q 8f1084e
pip install -e "$TP/LIBERO" --no-deps
if [ "$1" = "--curobo" ]; then
  pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  if [ ! -d "$TP/curobo/.git" ]; then git clone https://github.com/NVlabs/curobo.git "$TP/curobo"; fi
  git -C "$TP/curobo" checkout -q v0.8.0 2>/dev/null || true
  pip install -e "$TP/curobo" --no-build-isolation
fi
echo "done; now: python tools/doctor.py"
