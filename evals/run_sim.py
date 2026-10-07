"""Run the subgoal suite against the LIBERO simulator and say which ones finished.

    tools/on_gpu.sh 1 python evals/run_sim.py
    ... --only over_basket,lift --task 0 --out evals/last_run.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter.libero import LiberoAdapter  # noqa: E402
from evals.checks import (body_position, grasp_point, run_check,  # noqa: E402
                          tool_position)
from src.controller import runner  # noqa: E402
from src.controller.clearance import from_wrist, grasp_state  # noqa: E402
from src.controller.convert import load_directions, to_deltas  # noqa: E402
from src.executor.context import (ClearanceTracker, TablePlane, aim_depth,  # noqa: E402
                                  describe_aim_point,  # noqa: E402
                                  describe_capabilities, describe_robot,  # noqa: E402
                                  paint_aim_marker)  # noqa: E402
from src.executor.loop import ExecutorLoop, Observation  # noqa: E402
from src.executor.proposal import Proposer  # noqa: E402
from src.executor.supervision import MotionSupervisor  # noqa: E402
from vlms import client_for  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = yaml.safe_load(open(os.path.join(ROOT, "src", "prompts", "executor.yaml")))


TABLE = TablePlane()
TRACKER = ClearanceTracker()
# The aim spot drawn on the pictures the model sees, at the pixel the text names.
AIM_MARKER = os.environ.get("EXECUTOR_AIM_MARKER", "1") != "0"
# NO GATE IS PASSED IN: the whole chain -- batch, repeat, rotation, empty pick-up, skid,
# residual -- lives in ExecutorLoop.run.


def build_observer(adapter):
    """The same observation the web page builds, without the web page."""
    import base64
    import io

    from PIL import Image

    def observe() -> Observation:
        pose = adapter.tcp_pose()
        payload = adapter.frames()
        cameras = payload.get("cameras") or {}
        grasp = grasp_state(adapter.gripper_reading()).to_dict()
        # The same surface reading the page shows.
        clearance = None
        wrist = cameras.get("wrist")
        if wrist is not None:
            try:
                clearance = from_wrist(
                    wrist, pose, adapter.capabilities().control_point_offset_m).to_dict()
            except Exception:
                clearance = None
        clearance = TRACKER.update(clearance, pose)
        # The table height the robot was commissioned with, when it has one: measured once,
        # against a plane, rather than read per frame from whatever is under the tool.
        commissioned = adapter.capabilities().table_z_m
        surface = TABLE.update(clearance) if commissioned is None else commissioned * 1000.0
        depth = aim_depth(pose, clearance, surface)
        images, names, capture = [], [], time.time()
        for name, camera in cameras.items():
            blob = camera.get("rgb_jpeg")
            if not blob:
                continue
            image = np.asarray(Image.open(io.BytesIO(base64.b64decode(blob))).convert("RGB"))
            camera["image"] = image            # what locate() measures: no marker on it
            images.append(paint_aim_marker(image, camera, pose, depth) if AIM_MARKER
                          else image)
            names.append(name)
            capture = float(camera.get("timestamp") or capture)
        seen = (clearance or {}).get("mm")
        if seen is None:
            seen = (clearance or {}).get("tracked_mm")
        return Observation(images=images, names=names, capture_time=capture,
                           robot_text=describe_robot(pose, clearance, grasp), pose=pose,
                           grasp=grasp or {},
                           legend=describe_aim_point(cameras, pose, clearance,
                                                     surface_z_mm=surface),
                           clearance_mm=None if seen is None else float(seen),
                           clearance_seen=(clearance or {}).get("mm") is not None,
                           cameras=cameras)

    return observe


def make_loop(adapter, spec: Dict[str, Any], directions, schema,
              default_cycles: int = 5) -> ExecutorLoop:
    """The one place an :class:`ExecutorLoop` is built for an eval."""
    caps = adapter.capabilities()
    return ExecutorLoop(adapter=adapter,
                        proposer=Proposer(client_for("executor"), PROMPTS, schema,
                                          directions),
                        supervisor=MotionSupervisor(client_for("executor"), PROMPTS),
                        observe=build_observer(adapter),
                        capability_text=describe_capabilities(caps),
                        max_cycles=int(spec.get("max_cycles", default_cycles)),
                        gripper_only=bool(spec.get("gripper_only", False)))


def truth_probe(adapter, spec: Dict[str, Any]):
    """A list of ground-truth rows and the callable that appends one."""
    truth: List[Dict[str, Any]] = []

    def measure():
        """One row of ground truth: where the tool is relative to the target right now."""
        body = spec.get("body") or (spec.get("check") or {}).get("body")
        if not body:
            return
        try:
            check = spec.get("check") or {}
            here = tool_position(adapter)
            # measure against the place a gripper should go when the check names one --
            # a bowl's rim, a box's top -- and against the body centre otherwise
            part = check.get("part") if check.get("fn") == "at_grasp_point" else None
            there = grasp_point(adapter, body, part) if part else body_position(adapter, body)
            diff = (here - there) * 1000.0
            gap = float(np.linalg.norm(diff[:2]))
            camera = (adapter.frames(["wrist"]).get("cameras") or {}).get("wrist") or {}
            K = np.asarray(camera["intrinsic"], float)
            T = np.asarray(camera["cam2base"], float)
            pc = T[:3, :3].T @ (there - T[:3, 3])
            uv = K @ pc
            truth.append({"gap_mm": round(gap), "gap_3d_mm": round(float(np.linalg.norm(diff))),
                          "above_mm": round(float(diff[2])),
                          "tool_mm": [round(float(v) * 1000) for v in here],
                          "target_across_pct": round(float(uv[0] / uv[2]) / camera["width"] * 100),
                          "target_down_pct": round(float(uv[1] / uv[2]) / camera["height"] * 100)})
        except Exception as exc:
            truth.append({"error": str(exc)[:60]})

    return truth, measure


def run_one(adapter, spec: Dict[str, Any], directions, schema, verbose: bool,
            reset: bool = True) -> Dict[str, Any]:
    # Reset between subgoals when each is being scored on its own from the task's initial state.
    if reset:
        adapter.connect()
        TRACKER.reset()
    if spec.get("setup"):
        # a manual command run BEFORE the subgoal, unseen by the model: how a test puts the
        # robot somewhere deliberate -- 45 mm off the end of the box, so the first close
        # must miss and the recovery is what gets exercised
        runner.run(adapter, to_deltas({"actions": spec["setup"]}, adapter.capabilities()))
    start = {"tool": tool_position(adapter).copy()}
    body = (spec.get("check") or {}).get("body")
    if body:
        # where the object was, for checks that ask whether IT moved
        start["objects"] = {body: body_position(adapter, body).copy()}

    loop = make_loop(adapter, spec, directions, schema)
    truth, measure = truth_probe(adapter, spec)

    def on_event(event):
        # rows are taken when the model LOOKS, so row k is the state its k-th proposal was
        # made from; the final row, after the last motion, is the state that gets scored
        if event.get("event") == "observed":
            measure()

    started = time.monotonic()
    cycles = loop.run(spec["subgoal"], on_event=on_event, criterion=spec.get("criterion", ""))
    elapsed = time.monotonic() - started
    measure()

    claimed = any((c.proposal or {}).get("done") for c in cycles)
    check = run_check(spec["check"], adapter, start)
    row = {"name": spec["name"], "passed": check.passed, "check": check.as_dict(),
           "model_said_done": claimed, "cycles": len(cycles),
           "elapsed_s": round(elapsed, 1),
           "truth": truth, "detail": [c.as_dict() for c in cycles]}

    mark = "PASS" if check.passed else "FAIL"
    agree = "" if claimed == check.passed else \
        ("   <- the model said done but it was not" if claimed
         else "   <- finished without the model noticing")
    print("%-18s %-4s  %2d cycles  %5.1f s  %s%s"
          % (spec["name"], mark, len(cycles), elapsed, check.detail, agree))
    if verbose:
        for c in cycles:
            said = (c.proposal or {}).get("proposal") or {}
            print("    cycle %d: %s" % (c.index + 1, json.dumps(said.get("command"))))
            if said.get("assessment"):
                print("             %s" % said["assessment"][:110])
            for sv in c.supervision:
                print("             watch: %s -- %s" % (sv["decision"], sv.get("because", "")[:70]))
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0,
                        help="which of LIBERO's 50 saved initial states to start from")
    parser.add_argument("--only", default="", help="comma separated subgoal names")
    parser.add_argument("--spec", default="subgoals.yaml",
                        help="which subgoal file in evals/ to run")
    parser.add_argument("--out", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--camera-size", type=int, default=1024,
                        help="render size; the VLM has to identify objects in these pixels")
    args = parser.parse_args()

    spec = yaml.safe_load(open(os.path.join(ROOT, "evals", args.spec)))
    wanted = [n for n in args.only.split(",") if n]
    subgoals = [g for g in spec["subgoals"] if not wanted or g["name"] in wanted]
    if not subgoals:
        print("no subgoal matched {!r}".format(args.only))
        return 2

    sys.path.insert(0, os.path.join(ROOT, "web"))
    from server import action_schema
    directions = load_directions()
    schema = action_schema(directions)

    reachable, detail = client_for("executor").health()
    print("model: {}".format(detail))
    if not reachable:
        return 2

    adapter = LiberoAdapter(suite=args.suite or spec["suite"],
                            task_id=args.task if args.task is not None else spec["task_id"],
                            seed=args.seed, camera_size=args.camera_size)
    adapter.connect()
    print("task : {}\n".format(adapter.task_language))
    print("%-18s %-4s  %-9s %-7s %s" % ("subgoal", "", "cycles", "time", "what was measured"))

    rows: List[Dict[str, Any]] = []
    try:
        sequential = bool(spec.get("sequential"))
        # In a sequential file, an entry that names its own scene starts a NEW chain; the
        # entries after it inherit its state.
        skipping = False
        for index, entry in enumerate(subgoals):
            starts_chain = index == 0 or "suite" in entry or "task_id" in entry
            if sequential and skipping and not starts_chain:
                continue
            skipping = False
            suite = entry.get("suite", adapter.suite_name)
            task = int(entry.get("task_id", adapter.task_id))
            if (suite, task) != (adapter.suite_name, adapter.task_id):
                adapter.set_task(suite, task)
                print("task : {}".format(adapter.task_language))
            rows.append(run_one(adapter, entry, directions, schema, args.verbose,
                                reset=not sequential or starts_chain))
            if sequential and not rows[-1]["passed"] and not entry.get("continue_on_fail"):
                print("   ... the rest of this chain is skipped: it would run from a state "
                      "this subgoal was supposed to produce")
                skipping = True
    finally:
        adapter.close()

    passed = sum(1 for r in rows if r["passed"])
    print("\n{} of {} subgoals finished".format(passed, len(rows)))
    disagreed = [r["name"] for r in rows if r["model_said_done"] != r["passed"]]
    if disagreed:
        print("the model's own verdict disagreed with the measurement on: {}"
              .format(", ".join(disagreed)))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"rows": rows}, handle, indent=1)
        print("wrote {}".format(args.out))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
