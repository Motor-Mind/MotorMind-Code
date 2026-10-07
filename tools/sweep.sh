#!/bin/bash
# The full libero_pro sweep (400 cells: 200 tasks x seeds 0,1) as two lines, each with its own
# executor server, Qwen only. RESUMABLE: re-running the same command skips finished cells.
#   usage: tools/sweep.sh TAG            -> $STORM_RUNS/qwen-TAG-{A,B}
# env (defaults = configs/servers.example.yaml's layout):
#   EXEC_A / EXEC_B   executor URL per line          (http://127.0.0.1:8083/v1, :8082)
#   MONITOR           monitor + memory server URL    (http://127.0.0.1:8084/v1)
#   PLANNER           planner server URL             (http://127.0.0.1:8081/v1)
#   GPUS_A / GPUS_B   sim GPUs per line, round-robin (0,1 / 1,0)
#   STORM_RUNS        where runs go                  (<repo>/evals/runs)
#   EXTRA             more sweep_missions.py args    (e.g. "--min-free-mb 800 --reserve-mb 800")
#   PYTHON            interpreter                    (python)
set -e
T=${1:?TAG}; ROOT=$(cd "$(dirname "$0")/.." && pwd); RUNS=${STORM_RUNS:-$ROOT/evals/runs}
EXEC_A=${EXEC_A:-http://127.0.0.1:8083/v1}; EXEC_B=${EXEC_B:-http://127.0.0.1:8082/v1}
MONITOR=${MONITOR:-http://127.0.0.1:8084/v1}; PLANNER=${PLANNER:-http://127.0.0.1:8081/v1}
PY=${PYTHON:-python}
for u in "$EXEC_A" "$EXEC_B" "$MONITOR" "$PLANNER"; do
  [ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$u/models")" = 200 ] || { echo "server $u down"; exit 1; }
done
cd "$ROOT"
for spec in "A|0-49,100-149|$EXEC_A|${GPUS_A:-0,1}" "B|50-99,150-199|$EXEC_B|${GPUS_B:-1,0}"; do
  IFS='|' read -r name tasks exec gpus <<< "$spec"; O=$RUNS/qwen-$T-$name; mkdir -p "$O"
  env -u LIBERO_CONFIG_PATH -u VLM_API_KEY_FILE -u VLM_MODEL -u VLM_REASONING_EFFORT PYTHONUNBUFFERED=1 \
    QWEN_EXECUTOR_URL=$exec QWEN_PLANNER_URL=$PLANNER QWEN_MONITOR_URL=$MONITOR QWEN_MEMORY_URL=$MONITOR \
    setsid nohup $PY evals/sweep_missions.py --suite libero_pro --tasks $tasks --seeds 0,1 --jobs 2 \
    --gpus $gpus --timeout-min 12 --out "$O" --stop-on-success --budget-s 450 $EXTRA \
    >> "$O/sweep.log" 2>&1 < /dev/null &
done
echo "$(date +%H:%M) sweep $T -> $RUNS/qwen-$T-{A,B}; watch: pgrep -af 'sweep_mission[s].py'"
