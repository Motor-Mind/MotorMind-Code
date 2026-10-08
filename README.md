# MotorMind

**Scaffolding General Vision Language Models for Zero-Shot Robot Manipulation**

Bingxuan Li\*, Siqi Song\*, Yizhuo Wu\*, Jiarui Yao, Tong Zhang, Huan Zhang

University of Illinois Urbana-Champaign

[Paper](https://arxiv.org/pdf/2609.38078) · [Project page](https://motor-mind.github.io) 

![MotorMind](https://motor-mind.github.io/static/images/motormind/teaser.png)

MotorMind bridges VLM reasoning and robot control through **mid-level actions**, **continual
refinement** and **asynchronous scheduling**. It manipulates zero-shot, with no task-specific
training, both in simulation (Franka in LIBERO-PRO) and on a real xArm6. It uses no VLA as a tool,
no action experts, no coding agent, no motion planner or IK module, no SAM, and no skill API.

## How it works

![Method](https://motor-mind.github.io/static/images/motormind/method.png)

One frozen VLM takes five roles, and a deterministic **controller** turns its proposals into
motion:

| Role | Does | Code |
|---|---|---|
| Planner | splits the task into subgoals, and mends the plan after a failure | `src/planner/planner.py` |
| Executor | proposes short sequences of mid-level actions for the current subgoal | `src/executor/` |
| Monitor | checks the scene while a motion runs, and can cancel pending commands at an action boundary | `src/planner/monitor.py` |
| Verifier | judges whether a subgoal actually happened | `src/planner/verify.py` |
| Memory | keeps a short note of what has happened, updated in the background | `src/memory/` |

- **Mid-level actions.** The VLM proposes parameterized moves, rotations and gripper actions (see
  [the command language](#the-command-language)). The controller validates and executes them.
- **Continual refinement.** Fresh observations and measured feedback inform the next proposal.
  The outcome decides whether to advance, retry or replan.
- **Concurrent feedback.** The monitor and the memory run alongside execution, not between steps.

The same harness drives the simulated Franka and the real xArm6; only the adapter changes.

## Layout

| Path | What |
|---|---|
| `src/schema/` | the command language: five action types, and the direction words (`directions.yaml`) |
| `src/controller/` | command → `TcpDelta`s (`convert.py`), `plan()` / `run()` (`runner.py`), the `RobotAdapter` boundary (`types.py`) |
| `src/executor/` | the per-subgoal loop: propose actions, supervise them, locate objects from camera boxes and depth |
| `src/planner/` | mission planning, the scene monitor, and the verifier that judges each subgoal |
| `src/memory/` | the running memory note handed from one subgoal to the next |
| `src/prompts/` | prompts, as data |
| `vlms/` | the model client and the role → server table |
| `adapter/libero/` | the LIBERO backend (OSC_POSE servo loop) and the LIBERO-PRO suites |
| `adapter/xarm6/` | the xArm6 backend, its HTTP bridge client, and `mock_bridge.py` (the bridge with no robot) |
| `robot/` | the xArm6 model and limits (`robot.yaml`), per-rig measured profiles (`rigs/`), and `bridge_conformance.py` |
| `web/` | the operator page: manual commands, single subgoals and whole missions over live cameras, with STOP |
| `evals/` | mission runner, resumable sweeps, reports, and the subgoal suites |
| `tools/` | setup, model serving, GPU selection, the sweep script, scoring, rig calibration |
| `sim/libero/` | this repo's LIBERO scene definitions and initial states |
| `dynamic_libero_tasks/` | 20 LIBERO tasks with moving objects (conveyor and carousel) |

## Install

Needs Linux, an NVIDIA driver that ships EGL (offscreen MuJoCo rendering), and Docker with the
NVIDIA runtime for the model servers.

```bash
git clone https://github.com/Motor-Mind/MotorMind-Code.git && cd MotorMind-Code
conda env create -f environment.yml && conda activate motormind
tools/setup_third_party.sh            # LIBERO at the pinned commit, into third_party/
tools/setup_third_party.sh --curobo   # optional: also cuRobo, for the xArm6 motion pre-flight
```

**The simulator.** `sim/libero/` holds this repo's scenes and initial states, and the adapter
points LIBERO at them through `LIBERO_CONFIG_PATH`; your global `~/.libero/config.yaml` is left
alone. The meshes and textures are not in git. They are read from `sim/libero/assets` if you
copy them there, otherwise from the LIBERO install that `tools/setup_third_party.sh` made.

## Start the model servers

The roles above are served by OpenAI-compatible VLM servers. Each role is looked up by its own name, and the verifier uses the planner's server:

| Role | Does | Default |
|---|---|---|
| `planner` | the planner and the verifier | `:8081` |
| `executor` | the executor: proposes and supervises motions | `:8082` |
| `monitor` | the monitor | `:8081` |
| `memory` | the memory | `:8081` |

The reference model is Qwen3.8-Flash-Next-FP8 under SGLang, one TP=2 server per GPU pair:

```bash
tools/serve_qwen.sh 4,5 8081          # GPUS PORT [NAME]: starts a docker container
tools/serve_qwen.sh 6,7 8082
curl -s http://127.0.0.1:8081/v1/models   # wait until it answers
```

Each server fills its two GPUs, so the simulator needs a GPU of its own.

To point roles at other servers, copy `configs/servers.example.yaml` to `configs/servers.yaml` and
edit it, or set an environment variable. Lookup order is `QWEN_<ROLE>_URL`, then
`$STORM_SERVERS` or `configs/servers.yaml`, then the defaults above. A comma list in
`QWEN_<ROLE>_URL`, or a YAML list, gives a role several replicas.

## Check the machine

```bash
python tools/doctor.py              # packages, LIBERO scenes and meshes, GPUs and EGL, servers
python tools/doctor.py --no-gpu     # a machine that only drives remote servers
```

It lists what is missing and how to fix each item, and exits 1 if a run cannot start.

## Use it

### The operator page

```bash
# simulated Franka in LIBERO, rendering on CUDA GPU 0
tools/on_gpu.sh 0 python web/server.py --port 3008 --backend libero --suite libero_10 --task 0

# the real xArm6's protocol, with no robot
python adapter/xarm6/mock_bridge.py --port 18766 --token dev &
python web/server.py --port 3008 --bridge http://127.0.0.1:18766 --token dev
```

Open `http://127.0.0.1:3008` (it listens on 127.0.0.1; use `--host 0.0.0.0` or an SSH tunnel to
reach it remotely). The page has two tabs:

- **console**: type a command and see what it resolves to (the **Plan** panel) before running
  it, or hand the executor one subgoal in plain words (**Run subgoal**).
- **mission**: give a whole task. The planner, executor, monitor and memory run it end to end.

The header switches backend and task without a restart, and toggles cameras one by one. **STOP**
is always available.

`tools/on_gpu.sh N` sets the MuJoCo/EGL variables for CUDA GPU N. Use it for anything that renders
LIBERO, because the EGL device index is not the CUDA index on every host.

### The command language

```json
{
  "speed_mm_s": 10,
  "actions": [
    {"type": "move",    "direction": "forward",  "distance_mm": 80},
    {"type": "rotate",  "direction": "yaw_left", "angle_deg": 15},
    {"type": "gripper", "state": "close"},
    {"type": "wait",    "seconds": 2}
  ]
}
```

| `type` | Required | Optional |
|---|---|---|
| `move` | `direction` or `axis`, `distance_mm` | `stop_on_contact` |
| `rotate` | `direction` or `axis`, `angle_deg` | |
| `gripper` | `state`: `open` / `close` | `width_mm` |
| `home` | | |
| `wait` | `seconds` | |

- Directions come from `src/schema/directions.yaml`: `forward`, `backward`, `left`, `right`,
  `up`, `down`, and the rotations `yaw_left` / `yaw_right`, `pitch_up` / `pitch_down`,
  `roll_cw` / `roll_ccw`.
- `axis` is `[x, y, z]` in the robot base frame, which is the only frame there is.
- Distances and angles are magnitudes: write `backward`, not a negative `forward`.
- A push is a `move` with `"stop_on_contact": true`. It drives until the tool stops making
  headway, up to 250 mm.
- Options on an action or the whole command: `speed_mm_s`, `rot_speed_deg_s`, `stop_on_contact`,
  `note`.
- A bare list of actions, or one action object, is accepted too.

### One mission in simulation

```bash
tools/on_gpu.sh 0 python evals/run_mission.py --suite libero_10 --task 0 --budget 10
python evals/run_mission.py --task 8 --plan-only        # the plan only, no motion
```

The environment's own success predicate decides the result. Output goes to
`evals/runs/mission-<date>/task<N>_<k>.json` (`--out` to choose). `--record-dir DIR` also keeps
every frame, model call and event, plus an MP4 unless `--no-video` is given. Useful flags:

- `--seed` picks the LIBERO initial state.
- `--budget-s` sets the wall-clock limit.
- `--stop-on-success` stops as soon as the task is done.
- `--no-monitor` and `--no-memory` switch those roles off.

Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, and `libero_pro`
(200 tasks: the LIBERO-PRO perturbation variants of the four suites).

### A sweep, and its score

```bash
# any suite, tasks x seeds, resumable: rerun the same command to continue
python evals/sweep_missions.py --suite libero_10 --tasks 0-9 --seeds 0,1 --jobs 2 --gpus 0,1 \
    --out evals/runs/my-sweep
python evals/report_missions.py evals/runs/my-sweep           # one markdown table

# the full libero_pro sweep (400 cells) as two lines with their own executor servers
GPUS_A=0 GPUS_B=1 tools/sweep.sh mine          # -> evals/runs/qwen-mine-{A,B}
tools/score.py mine='qwen-mine-*'              # per-suite table, failures grouped by cause
```

`tools/sweep.sh` reads `EXEC_A`, `EXEC_B`, `PLANNER` and `MONITOR` for the server URLs (defaults
`:8083`, `:8082`, `:8081`, `:8084`). It refuses to start if any of them is down. A sweep
directory holds one `task<N>_seed<S>.json` and `.log` per cell, plus `sweep.log`, and
`REPORT.md` once it finishes. `$STORM_RUNS` moves `evals/runs`.

Stop a sweep with `kill -TERM <pid>` or Ctrl-C. A cell killed mid-run leaves no JSON, so the
resume runs it again.

### The real xArm6

The robot PC runs a bridge, an HTTP service in front of the arm, its gripper and its cameras.
**The [bridge documentation](adapter/xarm6/README.md) is its contract**: every route, header, unit and safety rule. Hand
it to whoever runs the robot PC.

`XARM_BRIDGE_TOKEN` is the access credential for the robot PC's bridge service. Obtain it from running the bridge and set it on the machine running MotorMind. MotorMind sends it
in the `X-Bridge-Token` HTTP header; the bridge rejects requests with a missing or incorrect
token. This is separate from any model API key and does not change when you change the
workspace scene. See the [bridge authentication rules](adapter/xarm6/README.md#3-safety-rules-the-bridge-enforces-non-negotiable) for
the protocol requirements.

```bash
# check a bridge against the contract (the robot PC can run this, stdlib only)
python robot/bridge_conformance.py --url http://127.0.0.1:18765 --token "$XARM_BRIDGE_TOKEN"
#   --gripper moves the jaws; --move and --extrinsic-check MOVE THE ARM

# describe your rig: copy the template, fill in the measured values, then check them
cp robot/rigs/TEMPLATE.yaml robot/rigs/my-rig.yaml
STORM_RIG=my-rig python tools/calibrate_rig.py --gripper   # prints the YAML lines that differ

# run
export XARM_BRIDGE_TOKEN=...          # from the robot PC, out of band; never committed
STORM_RIG=my-rig python web/server.py --port 3008 --bridge http://<robot-pc>:18765
```

`robot/robot.yaml` holds the arm's fixed values. A rig profile in `robot/rigs/` overrides them
with what was measured on one rig; `robot/rigs/x-arm6.yaml` is the rig MotorMind was fitted on, and
is the default. The bridge's contact detection is marked `trusted: false` until the robot PC
fixes it, so `stop_on_contact` (and therefore a push) is not reliable on the real arm.

#### Configuring your own workstation

A rig profile describes the robot, its workspace, its cameras and its bridge connection.
This configuration mechanism supports another xArm6 workstation using the compatible bridge
described above. A different robot model needs a suitable adapter or compatible implementation;
a YAML file alone does not provide that integration.

Copy `robot/rigs/TEMPLATE.yaml` to `robot/rigs/my-rig.yaml` and replace the example values with
your workstation's measurements. Set `bridge.url` to your bridge address and provide
`XARM_BRIDGE_TOKEN` before running the calibration check. `tools/calibrate_rig.py` reports
measurements and differences; it **does not update the YAML file**. Copy the applicable values
back into your profile. Its optional `--gripper` flag opens and closes the physical gripper.

For the same robot in a different scene, review these settings:

| What changed | What to update |
|---|---|
| Table height relative to the robot base | `workspace.table_z_mm`, in millimetres in the base frame |
| Required minimum tool height above the table | `workspace.floor_above_table_mm` |
| Maximum height of objects in the workspace | `workspace.max_object_height_mm`, used to estimate localization error when height has not been measured |
| Object positions only | Usually no profile change; positions are obtained from observations |
| Camera mounting or camera hardware | Recalibrate the relevant camera intrinsics/extrinsics on the bridge; also review `cameras.wrist.depth_min_range_m` and `depth_floor_band_share` |
| Tool or gripper | Update the corresponding `robot` fields and the bridge configuration |

The current template omits `workspace.floor_above_table_mm`. Add it explicitly to your profile
if you require a minimum tool height above the table; `robot/rigs/x-arm6.yaml` uses `10.0` mm
for the original workstation. Choose the value for your own setup.

Once the profile and bridge credentials are ready, run from the repository root:

```bash
# Uses robot/rigs/my-rig.yaml, including its bridge.url
STORM_RIG=my-rig python web/server.py --port 3008

# Alternatively, select a YAML file outside the repository
STORM_RIG=/absolute/path/my-rig.yaml python web/server.py --port 3008
```

No registration step is needed. The selected profile overrides `robot/robot.yaml` recursively
by key; it does **not** inherit from `robot/rigs/x-arm6.yaml`. Without `STORM_RIG`, `x-arm6` is
selected. The server's `--bridge` argument overrides the profile's bridge URL. Restart the
server after editing a profile; configuration is read when the adapter is initialized.

**Current limits:** not every scene assumption is configurable through YAML. Shared code still
contains object-height acceptance limits, localization thresholds and motion-clearance margins.
A missing or null `workspace.max_object_height_mm` falls back to a 160 mm assumption from
LIBERO.

### Moving objects (dynamic tasks)

`dynamic_libero_tasks/` has 20 tasks where the target rides a conveyor or a carousel until it is
grasped. On this branch the harness does not drive them. They run on their own:

```bash
tools/on_gpu.sh 0 python -m dynamic_libero_tasks.demo --only alphabet   # oracle interception -> MP4
python dynamic_libero_tasks/gen_tasks.py                                # regenerate the task files
```

## Environment variables

| Variable | Effect |
|---|---|
| `QWEN_<ROLE>_URL` | server(s) for a role (`PLANNER`, `EXECUTOR`, `MONITOR`, `MEMORY`) |
| `QWEN_<ROLE>_IMAGE_SIDE` | picture size a role sends (default 512; planner 768) |
| `STORM_SERVERS` | servers file in place of `configs/servers.yaml` |
| `STORM_RUNS` | where runs go (default `evals/runs`) |
| `STORM_RIG` | rig profile under `robot/rigs/` (default `x-arm6`) |
| `XARM_BRIDGE_TOKEN`, `MOTION_BRIDGE_URL` | the xArm6 bridge's token and URL |

## Citation

```bibtex
@article{li2026motormind,
  title={MotorMind: Scaffolding General Vision Language Models for Zero-Shot Robot Manipulation},
  author={Li, Bingxuan and Song, Siqi and Wu, Yizhuo and Yao, Jiarui and Zhang, Tong and Zhang, Huan},
  journal={arXiv preprint arXiv:2609.38078},
  year={2026}
}
```
