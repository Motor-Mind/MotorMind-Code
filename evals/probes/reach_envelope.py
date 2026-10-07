"""Where this arm actually stops going where it is sent, measured, not commissioned.

    tools/on_gpu.sh 1 \\
        python evals/probes/reach_envelope.py --task 2 --out evals/runs/reach-probe/scene2.json

Says whether `REACH_M = 0.76` (adapter/libero/adapter.py) is the arm's envelope or a gate
cutting inside it -- the plate sits 733-759 mm from the base in every base scene. It sends
the tool STRAIGHT THROUGH the adapter and the runner -- never through the executor, so
`fit_to_reach` never sees the move -- out to each radius at an approach
height, and records what the OSC did with every command:

* the radius the tool actually reached, and whether any command stalled: the adapter's own
  "blocked" or "skidded" abort, or a command that came back under three quarters of what it
  asked (a command under the adapter's 3 mm arrival tolerance is not sent: it moves nothing
  by design);
* a 20 mm lateral move each way at that station, which is the move seen to freeze near the edge;
* a descent to z = 30 mm at that radius, which is the other move that froze ("blocked:
  moved 0 mm along the axis" at 753-757 mm in task16_seed1), then the lateral pair again and
  20 mm further out and back at that height;
* how far the tool's orientation drifted from the pose it left home in, because the REACH_M
  comment cites rotation nobody asked for, not travel, as the sign of the wall.

A station is approached from the home radius at the approach height, never at z = 30: the
first run of this probe found a descent at the home radius (453 mm) stopped dead 40 mm above
the table and a radial move from there blocked at 0.2 mm -- the arm folded under itself, not
a wall -- so the low envelope is measured the way the harness meets it, from above.

Two lanes are probed: straight out along base +x, and the azimuth of the plate in this scene,
the thing the arm has to reach. Every `*_main` body's radius and azimuth is printed so a stop
can be told from a collision (the first run's "clear" lane at -40 degrees ran over the stove).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from adapter.libero import LiberoAdapter  # noqa: E402
from src.controller import runner, types  # noqa: E402
from src.controller.types import TcpDelta  # noqa: E402

RADII_MM = (740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0, 880.0)
#: the heights a station is approached at -- a carry, and a low approach -- and the grasp
#: height every station then descends to
APPROACH_MM = (150.0, 80.0)
LOW_MM = 30.0
#: one command's worth of travel on this adapter (max_translation_m at connect)
CHUNK_MM = 50.0
LATERAL_MM = 20.0
#: under this a command is inside the adapter's arrival tolerance and is not sent
LEAST_MM = 4.0
#: a command that comes back under this share of what it asked stalled
ARRIVED_SHARE = 0.75


def _delta(axis, mm: float, label: str) -> TcpDelta:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return TcpDelta(kind=types.TRANSLATE, axis=axis, magnitude=float(mm) / 1000.0,
                    speed=0.05, index=0, label=label)


def _radius_mm(pose) -> float:
    return float(np.hypot(pose.position_m[0], pose.position_m[1])) * 1000.0


def _drift_deg(start_rotation, pose) -> float:
    relative = np.asarray(pose.rotation, dtype=float) @ np.asarray(start_rotation, dtype=float).T
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _send(adapter, axis, mm: float, label: str, start_rotation, table_z_m: float) -> Dict[str, Any]:
    """One command, one row: what was asked, what the tool did, and whether it stalled."""
    before = adapter.tcp_pose()
    report = runner.run(adapter, [_delta(axis, mm, label)])
    result = report.results[0]
    after = adapter.tcp_pose()
    along = float(result.along_axis_m) * 1000.0
    stalled = bool(result.abort_reason) or along < ARRIVED_SHARE * float(mm)
    return {"label": label, "asked_mm": round(float(mm), 1), "along_mm": round(along, 1),
            "lateral_mm": round(float(result.lateral_m) * 1000.0, 1),
            "outcome": result.outcome, "abort": result.abort_reason or "",
            "r_before_mm": round(_radius_mm(before), 1), "r_after_mm": round(_radius_mm(after), 1),
            "z_after_mm": round((float(after.position_m[2]) - table_z_m) * 1000.0, 1),
            "drift_deg": round(_drift_deg(start_rotation, after), 1), "stalled": stalled}


def _obstacles(adapter) -> Dict[str, List[float]]:
    """Every object body on the table, xyz in the base frame, millimetres."""
    from evals.checks import body_position
    env = adapter._env.env
    out = {}
    for name in env.sim.model.body_names:
        if not name.endswith("_main") or any(w in name for w in ("table", "wall", "floor")):
            continue
        try:
            out[name] = [round(float(v) * 1000.0, 1) for v in body_position(adapter, name)]
        except Exception:
            continue
    return out


def probe_station(adapter, lane_deg: float, height_mm: float, radius_mm: float,
                  table_z_m: float) -> Dict[str, Any]:
    """Reset, go to the approach height, go out to the radius, then try the moves that froze."""
    adapter.connect()
    start = adapter.tcp_pose()
    start_rotation = np.asarray(start.rotation, dtype=float).copy()
    u = np.array([math.cos(math.radians(lane_deg)), math.sin(math.radians(lane_deg)), 0.0])
    side = np.array([-u[1], u[0], 0.0])
    rows: List[Dict[str, Any]] = []

    def go(axis, mm, label):
        row = _send(adapter, axis, mm, label, start_rotation, table_z_m)
        rows.append(row)
        return row

    def to_height(want_mm, label, limit=8):
        """Vertical chunks until the tool is at ``want_mm`` above the table, or stopped."""
        legs = []
        want = table_z_m + want_mm / 1000.0
        for _ in range(limit):
            dz = (want - float(adapter.tcp_pose().position_m[2])) * 1000.0
            if abs(dz) < LEAST_MM:
                break
            step = max(-CHUNK_MM, min(CHUNK_MM, dz))
            legs.append(go([0.0, 0.0, 1.0 if step > 0 else -1.0], abs(step), label))
            if legs[-1]["stalled"]:
                break
        return legs

    def lateral_pair(label):
        pair = [go(side, LATERAL_MM, label + " +"), go(-side, LATERAL_MM, label + " -")]
        return [pair[0]["along_mm"], pair[1]["along_mm"]], any(r["stalled"] for r in pair)

    to_height(height_mm, "to height")
    # ...then out along the lane until the radius is met, or the arm stops making headway
    out_rows = []
    for _ in range(12):
        here = np.asarray(adapter.tcp_pose().position_m, dtype=float)[:2]
        along = float(here @ u[:2])
        inside = (radius_mm / 1000.0) ** 2 - float(here @ here) + along * along
        left = (math.sqrt(max(0.0, inside)) - along) * 1000.0
        if left < LEAST_MM:
            break
        out_rows.append(go(u, min(CHUNK_MM, left), "out"))
        if out_rows[-1]["stalled"]:
            break
    at = adapter.tcp_pose()
    station = {"lane_deg": lane_deg, "height_mm": height_mm, "radius_mm": radius_mm,
               "reached_r_mm": round(_radius_mm(at), 1),
               "reached_z_mm": round((float(at.position_m[2]) - table_z_m) * 1000.0, 1),
               "out_stalled": any(r["stalled"] for r in out_rows),
               "drift_deg": round(_drift_deg(start_rotation, at), 1)}
    station["lateral_along_mm"], station["lateral_stalled"] = lateral_pair("lateral")
    # the descent to the grasp height at this radius, and the moves a grasp makes down there
    descent = to_height(LOW_MM, "down")
    low = adapter.tcp_pose()
    station["low_z_mm"] = round((float(low.position_m[2]) - table_z_m) * 1000.0, 1)
    station["low_r_mm"] = round(_radius_mm(low), 1)
    station["descent_stalled"] = any(r["stalled"] for r in descent)
    station["low_lateral_along_mm"], station["low_lateral_stalled"] = lateral_pair("low lateral")
    further = go(u, LATERAL_MM, "low out")
    back = go(-u, LATERAL_MM, "low in")
    station["low_radial_along_mm"] = [further["along_mm"], back["along_mm"]]
    station["low_radial_stalled"] = further["stalled"] or back["stalled"]
    station["low_drift_deg"] = round(_drift_deg(start_rotation, adapter.tcp_pose()), 1)
    station["rows"] = rows
    return station


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lanes", default="x,plate",
                        help="which of x, plate to probe, or a number of degrees")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    adapter = LiberoAdapter(suite=args.suite, task_id=args.task, seed=args.seed,
                            camera_size=256)
    adapter.connect()
    caps = adapter.capabilities()
    table_z = float(caps.table_z_m)
    home = adapter.tcp_pose()
    jaw = np.asarray(home.rotation, dtype=float) @ np.array([0.0, 1.0, 0.0])
    obstacles = _obstacles(adapter)
    plate = next((xy for name, xy in obstacles.items() if name.startswith("plate")), None)
    lanes: Dict[str, float] = {}
    for word in args.lanes.split(","):
        word = word.strip()
        if word == "x":
            lanes["x"] = 0.0
        elif word == "plate" and plate is not None:
            lanes["plate"] = round(math.degrees(math.atan2(plate[1], plate[0])), 1)
        elif word:
            lanes[word] = float(word)
    print("task  : {}".format(adapter.task_language))
    print("home  : tool {:.0f} mm out, {:.0f} mm above the table; jaws close along "
          "[{:+.2f}, {:+.2f}] in the base frame (x is out from the base)"
          .format(_radius_mm(home), (float(home.position_m[2]) - table_z) * 1000.0,
                  jaw[0], jaw[1]))
    print("reach : commissioned REACH_M {:.0f} mm, comfortable {:.0f} mm".format(
        caps.reach_m * 1000.0, (caps.comfortable_reach_m or 0.0) * 1000.0))
    for name, xy in sorted(obstacles.items()):
        print("body  : {:<36} r {:5.0f} mm  az {:+5.1f} deg  z {:+4.0f}".format(
            name, math.hypot(xy[0], xy[1]), math.degrees(math.atan2(xy[1], xy[0])),
            xy[2] - table_z * 1000.0))
    print("lanes : {}".format(lanes))

    stations: List[Dict[str, Any]] = []
    started = time.monotonic()
    try:
        for lane, deg in lanes.items():
            for height in APPROACH_MM:
                for radius in RADII_MM:
                    s = probe_station(adapter, deg, height, radius, table_z)
                    s["lane"] = lane
                    stations.append(s)
                    print("{:<5} z{:>3.0f} r{:>3.0f} -> r {:>5.1f} z {:>5.1f} {} lat {} {} "
                          "drift {:>4.1f} | down -> z {:>4.1f} r {:>5.1f} {} lat {} {} "
                          "radial {} {} drift {:>4.1f}".format(
                              lane, height, radius, s["reached_r_mm"], s["reached_z_mm"],
                              "STALL" if s["out_stalled"] else "ok   ",
                              s["lateral_along_mm"],
                              "STALL" if s["lateral_stalled"] else "ok   ",
                              s["drift_deg"], s["low_z_mm"], s["low_r_mm"],
                              "STALL" if s["descent_stalled"] else "ok   ",
                              s["low_lateral_along_mm"],
                              "STALL" if s["low_lateral_stalled"] else "ok   ",
                              s["low_radial_along_mm"],
                              "STALL" if s["low_radial_stalled"] else "ok   ",
                              s["low_drift_deg"]), flush=True)
    finally:
        adapter.close()
    print("{} stations in {:.0f} s".format(len(stations), time.monotonic() - started))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"task": adapter.task_language, "suite": args.suite,
                       "task_id": args.task, "seed": args.seed, "table_z_mm": table_z * 1000.0,
                       "home_r_mm": _radius_mm(home), "jaw_axis": [float(v) for v in jaw[:2]],
                       "obstacles": obstacles, "lanes": lanes, "stations": stations},
                      handle, indent=1)
        print("wrote {}".format(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
