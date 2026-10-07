#!/bin/bash
# Run a command with MuJoCo rendering on CUDA GPU N: sets MUJOCO_GL, the EGL device the driver
# reports for that GPU, and CUDA_VISIBLE_DEVICES (evals/sweep_missions.render_env).
#   usage: tools/on_gpu.sh N COMMAND...     e.g. tools/on_gpu.sh 1 python evals/run_sim.py
N=${1:?GPU}; shift; ROOT=$(cd "$(dirname "$0")/.." && pwd)
eval "$(cd "$ROOT" && ${PYTHON:-python} -c "
from evals.sweep_missions import render_env
print(' '.join('export {}={}'.format(k, v) for k, v in render_env($N).items()))")" || exit 1
exec "$@"
