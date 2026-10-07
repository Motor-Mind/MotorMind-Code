#!/usr/bin/env python
"""Measure what a rig profile needs that the bridge can tell, and compare it with the profile.

    python tools/calibrate_rig.py --rig x-arm6                  # read-only: /health, /state, /frames
    python tools/calibrate_rig.py --rig x-arm6 --gripper        # also close on nothing (jaws move)

Prints each measured value next to the profile's, and the YAML lines to paste into
robot/rigs/<rig>.yaml where they differ. Nothing is written. Values the bridge cannot measure
(pad length, depth minimum range, contact thresholds) are listed with how to measure them by hand
in robot/rigs/TEMPLATE.yaml.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapter.xarm6.adapter import load_config  # noqa: E402
from adapter.xarm6.bridge_client import BridgeClient  # noqa: E402


def dig(config, dotted):
    for key in dotted.split("."):
        config = (config or {}).get(key)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rig", default=None, help="profile name or path (default $STORM_RIG)")
    parser.add_argument("--url", default=None, help="bridge URL (default: the profile's)")
    parser.add_argument("--gripper", action="store_true", help="close the jaws on nothing")
    args = parser.parse_args()
    config = load_config(rig=args.rig)
    bridge = config.get("bridge") or {}
    client = BridgeClient(args.url or bridge.get("url"),
                          os.environ.get(bridge.get("token_env", "XARM_BRIDGE_TOKEN"), ""))
    health, state = client.health(), client.state()
    measured = {
        "workspace.table_z_mm": None if dig(health, "table.height_base_m") is None
        else round(float(dig(health, "table.height_base_m")) * 1000.0, 1),
        "robot.expected_bridge_tcp_offset_mm": dig(health, "robot.tcp_offset_mm"),
        "robot.gripper.full_open_gap_m": None,
        "robot.gripper.empty_close_raw": None,
    }
    if args.gripper:
        client.gripper("open")
        time.sleep(2.0)
        measured["robot.gripper.full_open_gap_m"] = dig(client.state(), "gripper.gap_m")
        client.gripper("close")
        time.sleep(2.0)
        measured["robot.gripper.empty_close_raw"] = dig(client.state(), "gripper.raw")
        client.gripper("open")
    wrist = dig(client.frames(cameras="wrist", size=int(bridge.get("frames_size", 256))),
                "cameras.wrist") or {}
    if wrist.get("depth_min_range_m") is not None:
        measured["cameras.wrist.depth_min_range_m"] = wrist["depth_min_range_m"]
    print("rig profile: {}  (arm state {}, error {})".format(
        config.get("rig"), state.get("state"), state.get("error_code")))
    for key, value in measured.items():
        have = dig(config, key)
        mark = "  " if value is None or value == have else "->"
        print("{} {:40} profile {!s:>28}  measured {!s}".format(mark, key, have,
                                                                 "-" if value is None else value))
    print("\n'->' rows differ: put the measured value in robot/rigs/<rig>.yaml under that key.")


if __name__ == "__main__":
    main()
