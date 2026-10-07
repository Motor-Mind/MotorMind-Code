"""One libero_10 task, planned and driven end to end, judged by the environment's predicate.

    tools/on_gpu.sh 1         python evals/run_mission.py --task 0 --budget 10
    python evals/run_mission.py --task 8 --plan-only        # the plan, no motion at all
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter.libero import LiberoAdapter  # noqa: E402
from evals.checks import body_extents, run_check, tool_position  # noqa: E402
from evals.run_sim import build_observer, make_loop  # noqa: E402
from src.controller.convert import load_directions  # noqa: E402
from src.executor.context import describe_capabilities  # noqa: E402
from src.planner.evidence import EvidenceLog  # noqa: E402
from src.memory.summarize import Summarizer  # noqa: E402
from src.planner.mission import (DEFAULT_MAX_REPLANS, MISSION_BUDGET_S,  # noqa: E402
                                 Mission)
from src.planner.monitor import SceneMonitor  # noqa: E402
from src.planner.planner import Planner, describe_plan  # noqa: E402
from src.planner.verify import Verifier  # noqa: E402
from src.recording import FPS as RECORD_FPS, MissionRecorder  # noqa: E402
from vlms import client_for, qwen  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLANNER_PROMPTS = yaml.safe_load(open(os.path.join(ROOT, "src", "prompts", "planner.yaml")))
MEMORY_PROMPTS = yaml.safe_load(open(os.path.join(ROOT, "src", "prompts", "memory.yaml")))


def health(roles: List[str]) -> bool:
    """Print every role's server before anything is rendered."""
    ok = True
    for role in roles:
        reachable, detail = client_for(role).health()
        print("{:<9} {}".format(role + ":", detail))
        ok = ok and reachable
    return ok


def _mm(vector) -> List[float]:
    return [round(float(v) * 1000.0, 1) for v in vector]


def _yaw_deg(xmat, base_rot) -> float:
    """Yaw about the base-frame z, in degrees, from MuJoCo's 3x3 body matrix."""
    matrix = np.asarray(xmat, dtype=float).reshape(3, 3)
    if base_rot is not None:
        matrix = np.asarray(base_rot, dtype=float).T @ matrix
    return round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 1)


def _goal_predicates(env) -> List[Dict[str, Any]]:
    """Every conjunct of the BDDL goal, evaluated one at a time by LIBERO's own predicate."""
    rows: List[Dict[str, Any]] = []
    for state in list(getattr(env, "parsed_problem", {}).get("goal_state") or []):
        row: Dict[str, Any] = {"predicate": str(state[0]).lower(),
                               "args": [str(a) for a in state[1:]]}
        try:
            row["true"] = bool(env._eval_predicate(list(state)))
        except Exception as exc:
            row["error"] = str(exc)[:80]
        rows.append(row)
    return rows


def _goal_names(env) -> List[str]:
    """Everything the goal talks about, in the order the BDDL names it."""
    names: List[str] = []
    for state in list(getattr(env, "parsed_problem", {}).get("goal_state") or []):
        for arg in state[1:]:
            if str(arg) not in names:
                names.append(str(arg))
    return names


def _is_site(env, name: str) -> bool:
    state = (getattr(env, "object_states_dict", None) or {}).get(name)
    return getattr(state, "object_state_type", "object") == "site"


def _articulation(env, name: str) -> Optional[Dict[str, Any]]:
    """The slide/hinge joints behind one goal name, with the ranges its predicate reads."""
    state = (getattr(env, "object_states_dict", None) or {}).get(name)
    if state is None:
        return None
    if getattr(state, "object_state_type", "object") == "site":
        holder = (getattr(env, "object_sites_dict", None) or {}).get(name)
        owner = env.get_object(getattr(state, "parent_name", "") or "")
    else:
        holder = env.get_object(name)
        owner = holder
    qpos: Dict[str, float] = {}
    model, data = env.sim.model, env.sim.data
    for joint in list(getattr(holder, "joints", None) or []):
        try:
            address = model.get_joint_qpos_addr(joint)
        except Exception:
            continue
        if isinstance(address, (tuple, list)):     # a free joint: pose, not articulation
            continue
        qpos[str(joint)] = round(float(data.qpos[address]), 4)
    if not qpos:
        return None
    ranges = (getattr(owner, "object_properties", None) or {}).get("articulation") or {}
    return {"qpos": qpos,
            "ranges": {str(k): [round(float(v), 4) for v in values]
                       for k, values in ranges.items() if values}}


def _task_names(env, extra=()) -> List[str]:
    """The objects of interest, then every movable object, then ``extra``, each once."""
    wanted: List[str] = []
    for name in (list(getattr(env, "obj_of_interest", None) or [])
                 + list((getattr(env, "objects_dict", None) or {}).keys()) + list(extra)):
        if str(name) not in wanted:
            wanted.append(str(name))
    return wanted


def _poses(env, to_base, base_rot):
    """Where every task-relevant body and goal region is, under its BDDL name."""
    objects: Dict[str, Any] = {}
    sites: Dict[str, Any] = {}
    model, data = env.sim.model, env.sim.data
    for name in _task_names(env, _goal_names(env)):
        try:
            if _is_site(env, name):
                position = data.get_site_xpos(name)
                sites[name] = {"xyz_mm": _mm(to_base(position))}
                continue
            body = (getattr(env, "obj_body_id", None) or {}).get(name)
            if body is None:
                body = model.body_name2id(name)
            objects[name] = {"xyz_mm": _mm(to_base(data.body_xpos[body])),
                             "yaw_deg": _yaw_deg(data.body_xmat[body], base_rot)}
        except Exception as exc:
            objects[name] = {"error": str(exc)[:60]}
    return objects, sites


def _finger_gap_mm(env) -> Optional[float]:
    """The jaw separation straight out of MuJoCo."""
    gripper = env.robots[0].gripper
    if isinstance(gripper, dict):
        gripper = list(gripper.values())[0]
    joints = list(getattr(gripper, "joints", None) or [])
    if len(joints) < 2:
        return None
    values = [float(env.sim.data.get_joint_qpos(joint)) for joint in joints[:2]]
    return round(abs(values[0] - values[1]) * 1000.0, 1)


def truth_row(env, to_base=None, base_rot=None, adapter_gripper=None) -> Dict[str, Any]:
    """One ground-truth reading of a LIBERO env: scoring data, never an input to a model."""
    to_base = to_base or (lambda v: v)
    row: Dict[str, Any] = {}
    for key, read in (("task_success", lambda: bool(env._check_success())),
                      ("goal", lambda: _goal_predicates(env)),
                      ("joints", lambda: {name: found for name in _goal_names(env)
                                          for found in [_articulation(env, name)] if found}),
                      ("gripper_mm", lambda: _finger_gap_mm(env))):
        try:
            row[key] = read()
        except Exception as exc:
            row[key] = None
            row.setdefault("errors", {})[key] = str(exc)[:80]
    try:
        row["objects"], row["sites"] = _poses(env, to_base, base_rot)
    except Exception as exc:
        row.setdefault("errors", {})["objects"] = str(exc)[:80]
    if adapter_gripper:
        row["adapter_gripper"] = adapter_gripper
    return row


def _adapter_gripper(adapter) -> Dict[str, Any]:
    """What the ADAPTER believes about the jaws, beside what the sim says."""
    from src.controller.clearance import grasp_state
    reading = adapter.gripper_reading()
    opening = None if reading.opening_m is None else round(reading.opening_m * 1000.0, 1)
    return {"gap_mm": opening, "holding": grasp_state(reading).holding}


def boxes_record(plan) -> Dict[str, Any]:
    """Where the planner said each object is, flat enough to score."""
    rows = []
    for step in plan.subgoals:
        if not step.target.strip():
            continue
        rows.append({"name": step.name, "target": step.target,
                     "boxes": dict(step.target_boxes)})
    return {"cameras": list(plan.boxed_on), "at": float(plan.boxed_at),
            "steps_naming_an_object": len(rows),
            "steps_with_a_box": sum(1 for row in rows if row["boxes"]),
            "rows": rows}


def plan_truth(adapter) -> Dict[str, Any]:
    """Where everything ACTUALLY is when the plan is written, and the cameras to project
    with."""
    env = adapter._env.env
    model = env.sim.model
    payload = adapter.frames().get("cameras") or {}
    cameras = {name: {"width": camera.get("width"), "height": camera.get("height"),
                      "intrinsic": camera.get("intrinsic"), "cam2base": camera.get("cam2base")}
               for name, camera in payload.items()}
    objects: Dict[str, Any] = {}
    for name in _task_names(env):
        try:
            body = (getattr(env, "obj_body_id", None) or {}).get(name)
            body_name = model.body_id2name(body) if body is not None else name
            lo, hi = body_extents(adapter, body_name)
            centre = adapter.base_from_world((lo + hi) / 2.0)
            footprint = adapter.base_from_world(
                np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]]))
            objects[name] = {
                "body": body_name,
                "centre_mm": _mm(centre),
                "footprint_mm": _mm(footprint),
                "extent_mm": _mm(hi - lo)}
        except Exception as exc:
            objects[name] = {"error": str(exc)[:80]}
    return {"cameras": cameras, "objects": objects}


def ground_truth_reader(adapter):
    """The env's goal predicate and where everything is, read after every subgoal."""
    def read() -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        try:
            row["tool_mm"] = [round(float(v) * 1000) for v in tool_position(adapter)]
        except Exception as exc:
            row["error"] = str(exc)[:120]
        try:
            row.update(truth_row(adapter._env.env,
                                 to_base=getattr(adapter, "base_from_world", None),
                                 base_rot=getattr(adapter, "_base_rot", None),
                                 adapter_gripper=_adapter_gripper(adapter)))
        except Exception as exc:
            row["error"] = str(exc)[:120]
        return row
    return read


def stop_when_the_env_says_so(adapter, mission, note: Dict[str, Any]):
    """EVALUATION ONLY: end the mission the moment the environment's own goal predicate goes
    true, instead of letting it run on to its own verdict and the clock.

    It reads the predicate off the simulator after every executed step, on the event stream
    the runner already listens to, and the one thing it does with what it reads is cancel the
    mission. Nothing it reads is written into an observation, a prompt or a gate -- the robot
    is never told -- and with `--stop-on-success` off the watcher is never installed at all.
    Truth is for scoring; this spends it only on when to stop the clock."""
    def watch(event: Dict[str, Any]) -> None:
        if note.get("stopped") or event.get("event") != "step_end":
            return
        try:
            check = run_check({"fn": "task_success"}, adapter, {})
        except Exception as exc:                 # a reading, never a dependency
            note["error"] = str(exc)[:120]
            return
        if check.passed:
            note.update({"stopped": True, "detail": check.detail})
            mission.cancel()
    return watch


def _tee(*listeners):
    """One event stream to several readers. The recorder reads it for which subgoal and
    cycle each frame belongs to, so it is told everything the printer is."""
    def watch(event):
        for listener in listeners:
            listener(event)
    return watch


class Printer:
    """One line per subgoal as it finishes, and the alerts that interrupted it beneath it."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.started = time.monotonic()
        self.cycles = 0
        self.attempt: Dict[str, Any] = {}

    def __call__(self, event: Dict[str, Any]) -> None:
        name = event.get("event")
        if name in ("planned", "replanned"):
            print("\nPLAN v{}{}".format(event["version"],
                                        "" if name == "planned" else
                                        "  (replanned: {})".format(event.get("trigger", ""))))
            for index, step in enumerate(event["plan"]["subgoals"], 1):
                print("  {}. {:<24} {} (done when: {}; up to {} motions)".format(
                    index, step["name"], step["subgoal"], step["criterion"],
                    step["max_cycles"]))
            for note in event.get("clamped") or []:
                print("     clamped: {}".format(note))
            print("")
            print("%-26s %-9s %-7s %-7s %s" % ("subgoal", "verdict", "cycles", "time", "why"))
        elif name == "subgoal_start":
            self.attempt = dict(event)
            self.started, self.cycles = time.monotonic(), 0
        elif name == "proposed":
            self.cycles += 1
            if self.verbose:
                said = (event.get("proposal") or {}).get("command")
                print("      cycle {}: {}".format(event.get("cycle"), json.dumps(said)))
        elif name == "alert":
            print("      ! {} ({}): {}".format(event["level"], event["finding"],
                                               event.get("because", "")[:90]))
        elif name == "note":
            if event.get("error"):
                print("      memory: no note ({})".format(event["error"][:80]))
            elif self.verbose:
                print("      memory v{}: {}".format(event["version"],
                                                    (event.get("note") or {}).get("summary", "")[:100]))
        elif name == "verdict":
            # The verdict names its own subject.
            label = event.get("name") or self.attempt.get("name", "?")
            mine = label == self.attempt.get("name")
            if mine and self.attempt.get("attempt", 1) > 1:
                label += " #{}".format(self.attempt["attempt"])
            print("%-26s %-9s %-7s %6s  %s"
                  % (label[:26], event.get("status") or "invalid",
                     self.cycles if mine else "-",
                     "%.1fs" % (time.monotonic() - self.started) if mine else "-",
                     (event.get("because") or "")[:60]))
        elif name == "finished":
            print("\nMISSION {} after {} cycles, {:.0f} s{}".format(
                event["status"].upper(), event["total_cycles"], event["elapsed_s"],
                "" if not event.get("error") else " -- " + event["error"]))


def out_path(args) -> str:
    """``evals/runs/mission-<date>/task<N>_<k>.json``, k the next free integer."""
    if args.out:
        return args.out
    folder = os.path.join(ROOT, "evals", "runs",
                          "mission-" + datetime.date.today().isoformat())
    os.makedirs(folder, exist_ok=True)
    if args.plan_only:
        return os.path.join(folder, "plan_only_task{}.json".format(args.task))
    k = 1
    while os.path.exists(os.path.join(folder, "task{}_{}.json".format(args.task, k))):
        k += 1
    return os.path.join(folder, "task{}_{}.json".format(args.task, k))


def write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1, default=str)
    print("wrote {}".format(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--seed", type=int, default=0,
                        help="which of LIBERO's 50 saved initial states to start from")
    parser.add_argument("--budget", type=int, default=10,
                        help="motions per subgoal; the planner is told it and clamped to it")
    parser.add_argument("--budget-s", type=float, default=MISSION_BUDGET_S,
                        help="the mission's whole wall clock in seconds, from the first plan "
                             "call; the executor's caps are safety rails under it")
    parser.add_argument("--max-replans", type=int, default=DEFAULT_MAX_REPLANS,
                        help="plan rewrites allowed; the mission's own default, passed "
                             "through, so the two cannot drift apart")
    parser.add_argument("--monitor-period", type=float, default=4.0)
    parser.add_argument("--no-monitor", action="store_true",
                        help="run without the scene monitor, to measure what it is worth")
    parser.add_argument("--no-memory", action="store_true",
                        help="run without the summariser; the planner gets no note")
    parser.add_argument("--plan-only", action="store_true",
                        help="take one observation, print the plan and stop")
    parser.add_argument("--camera-px", type=int, default=0,
                        help="render every camera at this many pixels instead of the "
                             "adapter's own default. Only useful together with "
                             "QWEN_<ROLE>_IMAGE_SIDE: the client downscales to the role's "
                             "size, so a bigger render alone reaches no model")
    parser.add_argument("--stop-on-success", action="store_true",
                        help="EVALUATION ONLY: end the mission as soon as the environment's "
                             "own goal predicate goes true. The predicate is read on the "
                             "event stream after every step and reaches no model, no "
                             "observation and no gate; the run file keeps the status the "
                             "harness itself had reached beside the `env_success` it ends on")
    parser.add_argument("--record-dir", default="",
                        help="keep every frame, every model call and every event of this "
                             "mission in this directory -- see src/recording.py")
    parser.add_argument("--record-fps", type=float, default=RECORD_FPS,
                        help="the MP4's nominal frame rate; frames.csv has the real times")
    parser.add_argument("--no-video", action="store_true",
                        help="record the calls and the events but assemble no video")
    parser.add_argument("--out", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    recorder = None
    if args.record_dir:
        recorder = MissionRecorder(args.record_dir, fps=args.record_fps,
                                   video=not args.no_video)
        # one process is one mission, so this catches every role's client, made where it likes
        qwen.RECORDER = recorder.on_call
        # The run JSON is written first and the videos last, in the `finally` below: a sweep's
        # SIGTERM has to become an exit that runs it, since a killed process flushes nothing.
        signal.signal(signal.SIGTERM, lambda number, frame: sys.exit(143))
        print("recording into {}".format(recorder.directory))

    roles = ["planner"] + ([] if args.plan_only else ["executor"]) \
        + ([] if args.plan_only or args.no_monitor else ["monitor"]) \
        + ([] if args.plan_only or args.no_memory else ["memory"])
    if not health(roles):
        return 2

    sys.path.insert(0, os.path.join(ROOT, "web"))
    from server import action_schema
    directions = load_directions()
    schema = action_schema(directions)

    sizes = {} if not args.camera_px else {"camera_size": args.camera_px,
                                           "side_camera_size": args.camera_px}
    adapter = LiberoAdapter(suite=args.suite, task_id=args.task, seed=args.seed, **sizes)
    if recorder is not None:
        adapter.on_frame = recorder.on_frame       # every picture it hands out, kept
    adapter.connect()
    task = adapter.task_language
    print("\ntask {}: {}".format(args.task, task))
    observe = build_observer(adapter)
    caps = adapter.capabilities()
    capability_text = describe_capabilities(caps)
    planner = Planner(client_for("planner"), PLANNER_PROMPTS, budget=args.budget)
    log, result = EvidenceLog(), None      # named before the try: the finally writes them out
    stopped_early: Dict[str, Any] = {}     # what --stop-on-success saw, if anything

    try:
        if args.plan_only:
            started = time.monotonic()
            result = planner.plan(task, observe(), capability_text)
            if not result.ok or result.plan is None:
                print("the planner never returned a plan: {}".format(result.error))
                write(out_path(args), {"task": args.task, "suite": args.suite,
                                       "variant": adapter.task_folder,
                                       "seed": args.seed, "language": task,
                                       "plan": result.as_dict()})
                return 1
            print("\n{}\n".format(result.plan.rationale.strip()))
            print(describe_plan(result.plan))
            for note in result.clamped:
                print("clamped: {}".format(note))
            print("\n{} subgoals, {} motions of budget, {:.0f} s".format(
                len(result.plan.subgoals),
                sum(s.max_cycles for s in result.plan.subgoals), time.monotonic() - started))
            record: Dict[str, Any] = {"task": args.task, "suite": args.suite,
                                      "variant": adapter.task_folder,
                                      "seed": args.seed, "language": task,
                                      "budget": args.budget, "plan": result.as_dict(),
                                      # where the planner said each object is...
                                      "boxes": boxes_record(result.plan)}
            try:                          # ...and where they actually are, for scoring only
                record["truth"] = plan_truth(adapter)
            except Exception as exc:
                record["truth"] = {"error": str(exc)[:160]}
            # what the plan cost, for the token/latency table: every attempt carries its own
            attempt = (result.as_dict().get("attempts") or [{}])[-1]
            record["cost"] = {"attempts": len(result.attempts),
                              "prompt_tokens": attempt.get("prompt_tokens"),
                              "completion_tokens": attempt.get("completion_tokens"),
                              "elapsed_s": round(result.elapsed_s, 2)}
            print("boxes: {} of {} step(s) naming an object carry one, in {}".format(
                record["boxes"]["steps_with_a_box"],
                record["boxes"]["steps_naming_an_object"],
                ", ".join(record["boxes"]["cameras"]) or "no cameras"))
            write(out_path(args), record)
            return 0

        read_truth = ground_truth_reader(adapter)
        tape: List[Dict[str, Any]] = []

        def taped_loop(sg):
            """The subgoal's executor loop, with a ground-truth row taken at every look."""
            loop = make_loop(adapter, sg.model_dump(), directions, schema)
            inner = loop.observe
            looks = [0]

            def observe_and_record():
                looks[0] += 1
                try:
                    tape.append({"subgoal": sg.name, "look": looks[0], **read_truth()})
                except Exception as exc:        # a record, never a dependency
                    tape.append({"subgoal": sg.name, "look": looks[0],
                                 "error": str(exc)[:120]})
                return inner()

            loop.observe = observe_and_record
            return loop

        mission = Mission(
            planner=planner,
            verifier=Verifier(client_for("planner"), PLANNER_PROMPTS),
            monitor=None,
            log=log,
            loop_factory=taped_loop,
            observe=observe,
            capability_text=capability_text,
            budget=args.budget,
            budget_s=args.budget_s,
            subgoal_s=(getattr(caps, "subgoal_median_s", None),
                       getattr(caps, "subgoal_verdict_s", None)),
            max_replans=args.max_replans,
            ground_truth=read_truth,
            summarizer=None if args.no_memory
            else Summarizer(client_for("memory"), MEMORY_PROMPTS))
        if not args.no_monitor:
            # Built after the mission because both of its callables are the mission's: the tee
            # it reads and the filter that decides what an alert does.
            mission.monitor = SceneMonitor(client_for("monitor"), PLANNER_PROMPTS,
                                           latest=mission.latest, on_alert=mission.on_alert,
                                           period_s=args.monitor_period)

        # Where everything stood before the robot moved: without it nothing in the file says
        # whether the knob started at 0 or the drawer started open.
        truth_start = read_truth()
        printer = Printer(args.verbose)
        listeners = [printer] + ([] if recorder is None else [recorder.on_event]) \
            + ([stop_when_the_env_says_so(adapter, mission, stopped_early)]
               if args.stop_on_success else [])
        watch = listeners[0] if len(listeners) == 1 else _tee(*listeners)
        result = mission.run(task, on_event=watch)
        if stopped_early.get("stopped"):
            # The harness's own last word is kept beside the ending, not overwritten by it:
            # the status it had reached (a cancel, from its side) and the last verdict any of
            # its own models gave, which is what says whether it KNEW the task was done.
            last = (result.subgoals or [{}])[-1]
            stopped_early.update(
                harness_status=result.status, elapsed_s=round(result.elapsed_s, 1),
                last_subgoal=last.get("name", ""),
                last_verdict=(last.get("verdict") or {}).get("verdict"))
            result.status = "env_success"
        success = run_check({"fn": "task_success"}, adapter, {})
        truth_end = read_truth()
        write(out_path(args), {"task": args.task, "suite": args.suite, "seed": args.seed,
                               # under `libero_pro` the suite is the aggregate; this is the one
                               # the task came from, which is what the report groups by
                               "variant": adapter.task_folder,
                               "language": task,
                               "budget": args.budget, "budget_s": args.budget_s,
                               "max_replans": args.max_replans,
                               "monitor": not args.no_monitor, "memory": not args.no_memory,
                               "mission": result.as_dict(), "task_success": success.as_dict(),
                               "stopped_on_success": dict(stopped_early),
                               "ground_truth_start": truth_start,
                               "ground_truth_end": truth_end,
                               "truth_tape": tape})
    finally:
        adapter.close()
        if recorder is not None:
            # The EvidenceLog is the mission's own record, which nothing else writes out.
            recorder.write_evidence(log.rows)
            recorder.write_json("notes.json", getattr(result, "notes", None) or [])
            kept = recorder.finish()
            print("recorded {} frames, {} calls, {} images, {} events -> {} ({:.1f} MB)"
                  .format(kept["frames"], kept["calls"], kept["images"], kept["events"],
                          kept["format"], kept["bytes"] / 1e6))
            for problem in kept.get("problems") or []:
                print("  recording: {}".format(problem))

    # `env_success` is this runner stopping the clock ON the predicate, so it agrees with it
    # by construction; what it does not claim is that the harness knew.
    agreed = (result.status in ("task_done", "env_success")) == success.passed
    print("\nTASK VERDICT: {}".format(success.detail))
    for entry in truth_end.get("goal") or []:
        print("   {} {}({})".format("TRUE " if entry.get("true") else "false",
                                    entry["predicate"], ", ".join(entry["args"])))
    if not agreed:
        print("   <- the mission said {} and the environment says {}".format(
            result.status, "SUCCESS" if success.passed else "not done"))
    return 0 if success.passed else 1


if __name__ == "__main__":
    sys.exit(main())
