# xArm6 bridge on the robot PC — what to build

You are the agent on the machine physically wired to the **UFACTORY xArm6**, its
**UFACTORY G1 gripper**, and **two calibrated RGB-D cameras** (a fixed "scene" camera looking
at the table, a "wrist" camera on the arm). This machine runs ONE thing for this project: a
small HTTP service, the **xArm bridge**, reachable from the GPU host over an SSH reverse
tunnel.

Nothing else from the GPU host is installed here, and nothing needs to be. The bridge is a
standalone program whose only dependencies are the xArm SDK, the camera SDK, numpy and
opencv. **This file is the complete contract.** Build to it; the acceptance test is §6.

```
 GPU host (storm_0918)                              this PC (robot side)
 ┌──────────────────────────────┐                    ┌──────────────────────────────────────┐
 │ web/server.py  :3008        │                    │ xArm bridge  http://127.0.0.1:18765  │
 │ semantic JSON -> delta TCP  │   ssh -R 18765     │   xArm-Python-SDK: joints, TCP, G1   │
 │ adapter/xarm6/bridge_client │◄───────────────────│   jobs: cartesian straight line,     │
 │   polls /state  10 Hz        │   JSON over HTTP   │         contact stop, heartbeat stop │
 │   polls /frames  2-3 Hz      │   (40-250 ms RTT)  │   cameras: aligned RGB+depth, K,     │
 │   one /cartesian job per move│                    │            cam2base per frame        │
 └──────────────────────────────┘                    └──────────────────────────────────────┘
```

The tunnel round trip is 40–250 ms, so the host never streams setpoints. It sends ONE
`/cartesian` job per motion (an absolute flange pose + a duration); the bridge plays it
locally as a straight line, stops it on contact when asked, and stops it when the host's
heartbeat stops.

## 0a. Punch list — measured against the live bridge, 2026-09-17

The bridge is up and the protocol is largely right: 73 of 74 read-only conformance checks pass,
all three cameras serve RGB + depth + K + `cam2base`, `/state` answers in 45 ms median and all
three cameras at 256 px in 44 ms, and the auth and `/stop` behaviour is exactly per §3. These
are the things still to fix, worst first.

1. **Contact fires on the acceleration ramp of every descent** (§5). A `move down 30 mm` over
   3 s stopped at waypoint 39 of 300, 1.17 mm in, fingertips ~290 mm above the table:
   `{"joint": 3, "start_current_a": -3.90, "current_a": -2.71, "delta_a": 1.19}`. **This makes
   `stop_on_contact` unusable today.** Logged descents on 2026-09-17 at 12.5 mm/s give the
   three numbers that fix it:

   | | i[1] shoulder | i[2] elbow |
   |---|---|---|
   | start-of-move transient, vs a trailing 0.5 s median | **0.74 A** (gone by t = 0.7 s) | 0.59 A |
   | free-air cruise, vs a trailing 0.5 s median | 0.23 A | **0.36 A** |
   | free-air, vs the current at job start (today's rule) | 0.83 A | **1.14 A** |
   | free-air drift over a 135 mm descent | 0.44 A | **0.99 A** |
   | **real table contact** | **2.70 -> 4.77 A** | **1.39 -> 2.40 A** |

   So: `ignore_first_s >= 0.8`, baseline = trailing median over ~0.5 s, threshold **1.0 A**.
   That sits above the 0.74 A transient and the 0.36 A cruise floor, and 2.7x below the
   weakest contact signal. A FIXED baseline fails at any offset — the elbow drifts 0.99 A
   over a long descent on its own.

   In that contact the current signature appeared at t = 2.81 s and **C31 fired at t = 3.24 s**,
   0.43 s and ~3 mm later: the current path is the faster detector once its baseline is right,
   with C31 as the backstop. `POST /stop` then recovered it in one call
   (`{"ok": true, "stopped": true, "recovered": true}` -> state 2, error 0) — the old rig's
   "/stop can't clear warnings" problem is gone.

1b. **The current feedback refreshes at ~5 Hz, not 100 Hz.** Polling `/state` at 25 Hz through
   the tunnel showed `joint_current_a` changing only 18 times in 3.7 s — median gap 195 ms, max
   228 ms. A 100 Hz detection loop reading a 5 Hz cache is a 5 Hz detector: at 12.5 mm/s that is
   up to 2.9 mm of travel after contact before the data can even show it. Raising the current
   sampling rate would sharpen every threshold above.
2. **`gripper.gap_m` is null**, and the closed endpoint is not the pads meeting. `/state`
   reports `gap_source: "unavailable"` with `gap_measured: false`; `/health.gripper_config`
   gives `open_raw: 850, close_raw: 230, width_measurement_method: null`. Observed
   2026-09-18: at `raw == close_raw == 230` the pads are **visibly apart** in the wrist view
   (their inner faces 10 px apart; no depth at that range, so no millimetres from here).

   Measure the pad gap with calipers at **both** endpoints, not just full open — the gap at
   `close_raw` is what says whether a thin object is being held or has simply slipped between
   the pads without moving the count. Two points give a linear raw-to-millimetres map, which
   turns every reading into a width. Report both, and set `width_measurement_method`.
3. **Say which indexing `contact_evidence.joint` uses.** It reported `"joint": 3` for a
   departure on `joint_current_a[2]`. Pick one convention, state it in `/health`.
4. **`wrist colour extrinsics: validation_failed`** — but an independent check here does not
   reproduce it. Segmenting one cube in all three views and back-projecting through each
   camera's own `cam2base` puts it at the same base-frame point within **17–27 mm**, wrist
   included; fitting the table plane in each camera agrees within **12 mm** (scene −69.7,
   wrist −62.1, side −57.7 mm at (0.5, 0), residuals under 1.6 mm RMS). So either the
   validation measures something stricter than metric consistency, or it is mis-firing.
   **Say what the check actually is and what it measured** before anyone refines a
   calibration that may already be good.
5. **Home pose is uncommissioned.** Pick one ~300 mm above the table, clear of the scene
   camera's view of the workspace, and more than one 50 mm job away from every reach limit.
6. **Table height**: measured from here at **−63 mm ± 6 mm** in the base frame, with 1.3–2.1°
   of tilt across the workspace. Adopt it or measure your own, and clear the readiness line.
7. **Close out the intrinsics drift**: `wrist ... differ from calibration by up to 1.570 native
   pixels`, `side ... 1.050`, "physical projection check pending". Small, but it is the last
   thing between `calibration_ready` and true.

## 0. What may already exist here

A bridge on this PC previously served `/health`, `/state`, `/trajectory/validate|start`,
`/trajectory/job/<id>` (+ `/heartbeat`), `/cartesian/validate|start|job`, `/capture`, `/stop`,
with an `X-Bridge-Token` header and `X-Motion-Confirmation: I_UNDERSTAND_ROBOT_WILL_MOVE` on
motion routes. Find it (`ps aux | grep -i bridge`, `systemctl list-units | grep -i xarm`,
`~/`, `/opt`) and read its source before writing anything: reuse its arm and camera code, its
token and its port. If it is gone, build from §5. Either way §6 is the gate.

## 1. Install

* Python 3.8+. `pip install xarm-python-sdk numpy opencv-python` plus the camera SDK
  (`pyrealsense2` for RealSense — **report the model and SDK you find**, §7).
* `autossh` (`apt install autossh`) for the tunnel.
* Run the bridge under `systemd` with `Restart=always` so it survives a crash or reboot, and
  log to a file. **Bind 127.0.0.1 only**; nothing on the LAN should reach it.

## 2. Tunnel

```
autossh -M 0 -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
        -R 18765:127.0.0.1:18765 <user>@<gpu-host> -p <ssh-port>
```

`127.0.0.1:18765` on the GPU host then reaches the bridge here. Run it as a second systemd
unit. Verify **both ends**: a listening port on the GPU host proves only that ssh is up — it
has happened before that the port listened and nothing answered. From the GPU host,
`curl -m 5 127.0.0.1:18765/health` must return JSON.

## 3. Safety rules the bridge enforces (non-negotiable)

1. Every route that can move anything (`/cartesian/start`, `/trajectory/start`, `/home`,
   `/gripper`) requires `X-Motion-Confirmation: I_UNDERSTAND_ROBOT_WILL_MOVE` → else **403**,
   and `X-Bridge-Token` → else **401**. Read routes require the token too.
   **`POST /stop` must ALWAYS work**: no confirmation header, never blocked by state. (A wrong
   token may be refused; nothing else may refuse it.)
2. **Heartbeat failsafe.** While a job runs the host POSTs `/<kind>/job/<id>/heartbeat` about
   every 0.3 s. If none arrives for `heartbeat_timeout_s` (advertise 2.0 in `/health`), stop the
   arm and finish the job `status: "stopped"`, `message: "heartbeat lost"`.
3. **Validate before moving.** `/cartesian/start` runs exactly the checks `/cartesian/validate`
   runs and refuses (400) what validate would reject. The `reason` string is shown to the
   operator, so make it specific ("target 612 mm from base, outside reach" beats "invalid").
4. **The arm's own collision detection stays on** (`set_collision_sensitivity`, default 3,
   overridable per job via `contact.collision_sensitivity`). After a collision error (C31) the
   bridge must recover on `/stop`: `clean_error()`, `motion_enable(True)`, `set_mode(...)`,
   `set_state(0)`, so the next job can run. **`/stop` must clear warnings too** — on the last
   rig session a warning-14 latch survived `/stop` and ended the session.
5. `/stop` aborts the running and queued jobs (`status: "aborted"`), stops the arm, clears
   errors, re-enables, answers `{"ok": true}`.
6. One job at a time. A second start while one runs is queued or refused (409); the host never
   submits two at once.

## 4. Two conventions that break everything if wrong

**Trap 1 — the two sides must agree on what `tcp_pose` points at, and it must not drift.**
The control point is the tool point **172 mm along the flange +z**. As of the 2026-09-17
rebuild, **this side keeps the arm's tool offset at zero and the host adds the 172 mm itself**:
call `set_tcp_offset([0,0,0,0,0,0])` at startup, report `tcp_offset_mm: [0, 0, 0, 0, 0, 0]` in
`/health`, and every `target_pose` the host posts is a **flange** pose.

This moved once already — the previous bridge configured 172 mm on the arm and reported
`tcp_pose` at the tool point. Same control point, opposite side of the wire. **Whichever side
holds the offset, the two must never both hold it**: doubled, every descent aims 172 mm deep.
So **tell the GPU side before you change it**. The host compares `/health.tcp_offset_mm`
against what it plans for at startup and before every run, and refuses on a mismatch, because
a changed offset moves the reference of every motion while nothing else looks any different.

**Trap 2 — which optical frame `cam2base` belongs to.** You say the cameras are calibrated;
this is the part "calibrated" does not pin down. Depth must be **aligned to the colour image**
(same pixel grid, e.g. RealSense `rs.align(rs.stream.color)`), `intrinsic` must be the COLOUR
stream's K at the served resolution, and `cam2base` must be the pose of the **colour** camera's
optical frame (OpenCV: x right, y down, z forward, metres) in the arm's base frame. The depth
sensor's optical centre sits 1–5 cm from the colour one; mixing them is the classic error.
For the **wrist** camera `cam2base` changes with every arm move: compute it per request as
`cam2base = T_base_flange(now) @ T_flange_cam`; never cache it.

Units everywhere: joints **radians**; `tcp_pose` **mm + radians**, `[x, y, z, roll, pitch, yaw]`
with `R = Rz(yaw)·Ry(pitch)·Rx(roll)`, exactly `arm.get_position(is_radian=True)`; depth
`uint16 × depth_scale` = metres (0 = no return); extrinsics metres; `gap_m` metres; timestamps
`time.time()` seconds.

## 4b. Motion caps to enforce (measured on this rig, 2026-09-16)

Per Cartesian job: translation **≤ 50 mm** total; rotation **≤ 0.1 rad (5.7°) per roll/pitch/yaw
component**; peak translation speed **20 mm/s** (a cosine profile → ~13 mm/s mean); joint speed
≤ 1.5 rad/s. Joint ranges to enforce at validate: J2 −117…+116°, J3 −219…+10°, J4 ±360°,
J5 −97…+180°, J6 ±360°. The host chunks longer motions into several jobs and stays just inside
these numbers; the bridge is the backstop. A target outside the joint ranges must come back as
an IK refusal at **validate** — if it only shows up once the servo stream hits it, the
controller parks the arm (warning 14) and the session is over.

## 5. The protocol

All bodies and replies are JSON. Joint arrays have `axis_count` (= 6) entries, joint1..joint6.
`<kind>` ∈ `cartesian` | `trajectory` | `home`; all kinds share the job/heartbeat shape.

### `GET /health`
```json
{"ok": true, "backend": "xarm-sdk",
 "robot": {"connected": true, "axis_count": 6, "mode": 1, "state": 2, "error_code": 0,
           "tcp_offset_mm": [0, 0, 0, 0, 0, 0]},
 "cameras": ["scene", "wrist"], "heartbeat_timeout_s": 2.0,
 "limits": {"max_translation_mm": 50, "max_rotation_rad": 0.1, "max_speed_mm_s": 20,
            "min_duration_s": 0.2, "max_duration_s": 120},
 "calibration_ready": true}
```

### `GET /state`  (polled at 10 Hz — keep it under 20 ms; cache SDK reads on a thread)
```json
{"joint_rad": [0.0, -0.042, -1.165, 0.0, 1.264, 0.0],
 "joint_vel_rad_s": [0, 0, 0, 0, 0, 0],
 "tcp_pose": [459.6, 0.0, 403.0, 3.1416, 0.057, 0.0],
 "gripper": {"gap_m": 0.0889, "moving": false, "command": "open"},
 "joint_current_a": [0.1, 0.9, 0.4, 0.1, 0.1, 0.0],
 "collision": false, "error_code": 0, "state": 2, "timestamp": 1757800000.123,
 "axis_count": 6, "job": null}
```
* `joint_rad` = `get_servo_angle(is_radian=True)`; zeros are accepted for `joint_vel_rad_s` if
  the SDK gives none. `joint_current_a` = `arm.currents`, the contact sensor for
  `stop_on_contact`. `collision` = true while the collision error (31) or the SDK flag is up.
  `state` = the xArm state int (0 moving, 2 ready, 4 stopped/error). `job` = running job id or
  null.
* `gripper.gap_m` = gap between the INNER pad faces, 0 closed … ~0.0889 fully open. **Measure
  the real full-open gap with calipers** and report it (§7); a geometry estimate is not enough,
  the host's grasp logic settles on this width.

### `GET /frames?cameras=scene,wrist&size=256`
```json
{"cameras": {
  "scene": {"rgb_jpeg": "<base64 JPEG>", "depth_png16": "<base64 16-bit greyscale PNG>",
            "depth_scale": 0.001, "intrinsic": [[fx,0,cx],[0,fy,cy],[0,0,1]],
            "cam2base": [[r,r,r,tx],[r,r,r,ty],[r,r,r,tz],[0,0,0,1]],
            "width": 256, "height": 256, "timestamp": 1757800000.1},
  "wrist": {"...": "same shape"}}}
```
* `size` absent → native resolution. `size=N` → the centre **square** crop (640×480 → central
  480×480) resized to N×N, with the intrinsic adjusted exactly: `cx -= x0`, `cy -= y0` for the
  crop, then fx, fy, cx, cy all `*= N/480`. Resize depth NEAREST (never average two surfaces),
  RGB INTER_AREA.
* Depth aligned to colour (Trap 2), uint16 in `depth_scale` units, 0 = no return. PNG level 1,
  JPEG quality ~90. Both cameras at 256 px ≈ 60 kB; serve in < 100 ms. Grab on a background
  thread and serve the latest pair — **never block a request on a camera grab**.
* Unknown camera name → 404.
* If the scene camera still stands low at the front-right: please raise it to look down at
  ~45° roughly in line with the base +x axis. From the low angle, "forward" and "left" both land
  on the image's horizontal axis, and an image offset cannot name a robot direction.

### `POST /cartesian/validate` and `POST /cartesian/start`  (start: + token + confirmation)
```json
{"target_pose": [x_mm, y_mm, z_mm, roll, pitch, yaw], "duration": 3.0,
 "goal_id": "g12", "stop_on_contact": true,
 "contact": {"joint_current_delta_a": 0.6, "collision_sensitivity": 3}}

validate → {"valid": true, "duration_s": 3.0, "translation_mm": 40.0, "rotation_rad": 0.0}
validate → {"valid": false, "reason": "translation 71.2 mm over the 50 mm limit"}
start    → {"id": "b3fa5abd8ff2", "status": "queued"}
```
A straight line of the **flange** to `target_pose` over `duration` seconds. **This is the route
the demo uses** — one job per motion, `duration` = distance / requested speed. Refuse (400) an
unreachable target, a duration outside `min_duration_s..max_duration_s`, an unreachable
orientation, or anything past §4b. `contact` is `{}` when `stop_on_contact` is false.

With `stop_on_contact: true`: watch the joint currents and, when one departs from its baseline
by more than `joint_current_delta_a`, or the collision flag rises, stop at once and finish
`status: "stopped"`, `contact: true`,
`contact_evidence: {"source": "joint_current", "joint": 2, "delta_a": 0.71}` (`source` required,
the rest free-form). Two things decide whether this works:

**The sign.** On a descent the table takes the arm's weight and the J2/J3 gravity current
*falls*. Compare the absolute departure, not only a rise — an earlier bridge only looked for a
rise and missed every table touch.

**The baseline must not be the current at job start.** A stationary arm holds a different
current from a moving one, so the first fraction of a second of any move looks exactly like a
contact. Measured on this rig, 2026-09-17: a `move down 30 mm` over 3 s, `stop_on_contact:
true`, `joint_current_delta_a: 0.6`, stopped at **waypoint 39 of 300** with
`{"joint": 3, "start_current_a": -3.90, "current_a": -2.71, "delta_a": 1.19}` — the elbow's
gravity load simply dropped as the descent began. The arm had travelled 1.17 mm (the cosine
profile predicts 1.23 mm at waypoint 39, so this was the acceleration ramp) and the fingertips
were **~290 mm above the table**. It cannot have touched anything.

So: **ignore contact for the first `contact.ignore_first_s` of a job (default 0.5 s), and take
the baseline from the end of that window, not from the stationary arm.** Honour
`ignore_first_s` in the `contact` body; report in `contact_evidence` which baseline was used.
Better still, baseline against a short rolling median of the moving current, so a slow
gravity-load change over a long descent does not creep past the threshold either.

**Index the `joint` field the same way as the array.** This rig reported `"joint": 3` for a
departure on `joint_current_a[2]` — 1-based in the evidence, 0-based in the array. Pick one,
say which in `/health`, and keep to it; an operator reading the evidence against the array is
otherwise looking at the wrong joint.

### `GET /cartesian/job/<id>`
```json
{"id": "b3fa5abd8ff2", "kind": "cartesian", "status": "running",
 "progress": {"fraction": 0.41}, "contact": false, "contact_evidence": null,
 "message": "", "elapsed_s": 1.65}
```
`status` ∈ `queued | running | complete | stopped | failed | aborted`. `stopped` = the bridge
stopped it (contact → `contact: true`; heartbeat → `message: "heartbeat lost"`); `failed` = the
arm faulted (error code in `message`); `aborted` = `/stop`. Unknown id → 404. Keep finished
jobs queryable for ≥ 60 s. A `stopped` job leaves the arm in state 4; the host calls `/stop` to
recover before the next motion.

### `POST /cartesian/job/<id>/heartbeat` → `{"ok": true}`  (404 for an unknown id)

### `POST /home {"q": [...6...]}`  (q optional; token + confirmation) → `{"id": ..., "status": "queued"}`
A slow planned move to the configured home pose (`set_servo_angle(..., speed=slow, wait=False)`
is fine). Polled at `/home/job/<id>` like any job. **Choose the home pose** (§7): above every
object, out of the scene camera's view of the workspace, joints away from limits, and — this
one bit us — *about 300 mm above the table and more than one 50 mm job away from any reach
limit*.

### `POST /gripper {"command": "open" | "close"}`  (token + confirmation) → `{"ok": true, "command": "close"}`
Non-blocking. `/state.gripper.moving` is true until the jaws stop (or ~1.5 s if the G1 reports
nothing); `gap_m` follows. Closing on an object must stop on the object, not fault.

### `POST /stop` → `{"ok": true}` — §3 rules 4 and 5.

### `POST /trajectory/validate|start`, `GET /trajectory/job/<id>` (optional, keep if it exists)
`{"points": [{"t": 0.0, "q": [...6...]}, ...]}`, `t` increasing at 20 Hz. Kept for the operator's
joint jog and as the execution path for cuRobo-planned trajectories later. Validate: joint
count, increasing `t`, first point within ~0.1 rad of the current joints, joint velocity ≤ 1.5
rad/s, joint limits, non-empty. Play the points honouring their `t` (a 4 s trajectory takes 4 s)
— servo mode (`set_mode(1)`, `set_state(0)`, `set_servo_angle_j` from a real-time thread,
interpolated to the servo period) is the intended path. Hold the last point at the end.

### `POST /capture {"session": "hand-eye"}`  (token)
Saves a native RGB-D pair from every camera plus the current `/state` to disk, answers with the
paths. Used to re-check the extrinsics.

## 6. Acceptance

Run the checker (`robot/bridge_conformance.py` in the host repo; stdlib only, copy it over) **on this machine**, against the bridge, no tunnel needed:

```
python3 bridge_conformance.py --url http://127.0.0.1:18765 --token "$XARM_BRIDGE_TOKEN"            # read-only
#   --expect-tcp-offset-mm 0,0,0,0,0,0 --host-grip-offset-mm 172  matches the host's config
python3 bridge_conformance.py --url http://127.0.0.1:18765 --token "$XARM_BRIDGE_TOKEN" --gripper  # jaws move
python3 bridge_conformance.py --url http://127.0.0.1:18765 --token "$XARM_BRIDGE_TOKEN" --move     # ARM MOVES: stand at the E-stop
```

`--move` homes the arm, runs a 20 mm Cartesian move out and back along each base axis, then
starts a job and stops heartbeating on purpose to prove the failsafe. **Every line must be
PASS.** WARN lines are things to explain in the report; FAIL lines are bugs.

**Then one extra check that "calibrated" does not cover** (this is the cheap way to catch a
transposed or wrong-frame extrinsic before it costs a run). With the gripper in view of both
cameras:

```
python3 bridge_conformance.py --url ... --token ... --extrinsic-check
```
or by hand: command a **+60 mm base-frame X** move, then **+60 mm Y**, then **−60 mm Z**,
capturing a scene and wrist frame before and after each. For each move report **(a)** the
measured TCP delta from `/state`, **(b)** the pixel displacement of the gripper in the scene
image, and **(c)** the same in the wrist image, and **(d)** the same delta computed by
projecting the TCP position through `intrinsic @ inv(cam2base)` before and after. (d) must match
(b)/(c) within a few pixels. If it does not, `cam2base` refers to the wrong optical frame, or
its rotation is transposed, or it maps base→cam instead of cam→base.

## 7. What to report back

1. Camera models and SDK; native resolution and FPS; how depth is aligned to colour.
2. The **token** (share out of band, not in a file), the port, `heartbeat_timeout_s`.
3. The **home pose**: 6 joint values in radians, and why it was chosen.
4. **Table**: height in the base frame (a table below the base plate reads negative z), its
   x/y extent in base coordinates, and where objects will sit.
5. The **validate limits** you implemented (velocity, start error, joint limits, min/max
   duration, translation/rotation caps).
6. Scene `cam2base` and wrist `T_flange_cam`: the 4×4s, how they were calibrated, which optical
   frame they refer to, and the §6 extrinsic-check output. `/health` currently reports both as a
   `rough_estimate` with `calibration_ready: false` — say what "rough" means here (hand-measured
   from the CAD? a hand-eye solve with how many poses and what residual?) and roughly how many
   **degrees** the rotation could be off by. The rotation is the part that matters to the host:
   a camera-frame command uses only `cam2base`'s 3×3, so a translation that is 2 cm out costs
   nothing, while a rotation 10° out sends every camera-frame motion 10° off.
7. `calibration_ready` currently also lists a provisional home pose and an unmeasured table
   height. Neither blocks the host, but both are in §7's list for a reason: pick a home that is
   ~300 mm above the table and more than one 50 mm job away from any reach limit, and measure
   the table height so the reachability pre-flight can include the table.
8. Measured: `/state` latency, `/frames?size=256` latency, the G1 full-open gap in mm (calipers),
   the servo rate you stream at, and what `stop_on_contact` did when a finger pad touched the
   table — **which joint's current moved, in which direction, and by how much** (this sets
   `joint_current_delta_a`, default 0.6 A).
9. The full output of the conformance runs, including `--extrinsic-check`.
