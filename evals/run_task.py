"""Run a whole libero_10 task as a chain of subgoals and report what actually happened.

    tools/on_gpu.sh 1         /path/to/python evals/run_task.py --task 0 --out evals/task0.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapter.libero import LiberoAdapter  # noqa: E402
from evals.chain_spec import expand, load  # noqa: E402
from evals.checks import run_check, tool_position  # noqa: E402
from evals.run_sim import make_loop, truth_probe  # noqa: E402
from src.controller.convert import load_directions  # noqa: E402
from vlms import client_for  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_subgoal(adapter, entry, directions, schema, verbose) -> Dict[str, Any]:
    start = {"tool": tool_position(adapter).copy()}
    # One construction path with run_sim.run_one -- the gate, gripper_only and the truth probe
    # all come from there.
    loop = make_loop(adapter, entry, directions, schema, default_cycles=6)
    truth, measure = truth_probe(adapter, entry)

    # Probe the subgoal's own check once per cycle.
    trace: List[float] = []

    def on_event(event):
        if event.get("event") != "observed":
            return
        measure()
        probe = run_check(entry["check"], adapter, start)
        if probe.measured is not None:
            trace.append(round(probe.measured, 1))

    started = time.monotonic()
    cycles = loop.run(entry["subgoal"], on_event=on_event, criterion=entry.get("criterion", ""))
    elapsed = time.monotonic() - started
    measure()

    check = run_check(entry["check"], adapter, start)
    claimed = any((c.proposal or {}).get("done") for c in cycles)
    row = {"name": entry["name"], "chain": entry["chain"], "step": entry["step"],
           "passed": check.passed, "check": check.as_dict(), "trace": trace,
           "truth": truth,
           "model_said_done": claimed, "cycles": len(cycles),
           "elapsed_s": round(elapsed, 1), "detail": [c.as_dict() for c in cycles]}
    if verbose:
        row["commands"] = [json.dumps(((c.proposal or {}).get("proposal") or {}).get("command"))
                           for c in cycles]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0,
                        help="which of LIBERO's 50 saved initial states to start from")
    parser.add_argument("--limit", type=int, default=0, help="only the first N subgoals")
    parser.add_argument("--only", default="", help="comma separated subgoal names")
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="override every selected subgoal's motion budget, to measure "
                             "whether a shape fails or only runs out")
    parser.add_argument("--chains", default="",
                        help="comma separated chain names to run in place of each task step's "
                             "own list, e.g. --chains take to run only the one-subgoal grasp")
    parser.add_argument("--out", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    spec = load()
    if args.chains:
        # The task's steps -- which object, which target -- are kept; only the chains run
        # against them change, so one primitive can be measured on every task without a
        # second copy of the task table.
        for step in spec["tasks"][args.task]["steps"]:
            step["chain"] = [c for c in args.chains.split(",") if c]
    entries = expand(spec, args.task)
    wanted = [n for n in args.only.split(",") if n]
    if wanted:
        entries = [e for e in entries if e["name"] in wanted]
    if args.limit:
        entries = entries[:args.limit]
    if args.max_cycles:
        for entry in entries:
            entry["max_cycles"] = args.max_cycles
    if not entries:
        print("no subgoal selected")
        return 2

    sys.path.insert(0, os.path.join(ROOT, "web"))
    from server import action_schema
    directions = load_directions()
    schema = action_schema(directions)

    reachable, detail = client_for("executor").health()
    print("model: {}".format(detail))
    if not reachable:
        return 2

    adapter = LiberoAdapter(suite=spec["suite"], task_id=args.task, seed=args.seed)
    adapter.connect()
    print("task {}: {}".format(args.task, adapter.task_language))
    print("{} subgoals\n".format(len(entries)))
    print("%-44s %-5s %-7s %-6s %s" % ("subgoal", "", "cycles", "time", "what was measured"))

    rows: List[Dict[str, Any]] = []
    failed = False
    try:
        for entry in entries:
            row = run_subgoal(adapter, entry, directions, schema, args.verbose)
            row["after_failure"] = failed
            failed = failed or not row["passed"]
            rows.append(row)
            mark = "PASS" if row["passed"] else "FAIL"
            tail = "  [after an earlier failure]" if row["after_failure"] else ""
            print("%-44s %-5s %2d      %5.1fs  %s%s"
                  % (row["name"][:44], mark, row["cycles"], row["elapsed_s"],
                     row["check"]["detail"], tail))
            if row["trace"]:
                print("%-44s          per cycle: %s" % ("", " -> ".join(str(v) for v in row["trace"])))
        success = run_check({"fn": "task_success"}, adapter, {})
    finally:
        adapter.close()

    clean = [r for r in rows if not r["after_failure"]]
    passed = sum(1 for r in clean if r["passed"])
    print("\n%d of %d subgoals passed before the first failure (%d run in total)"
          % (passed, len(clean), len(rows)))

    by_chain = defaultdict(lambda: [0, 0])
    for r in clean:
        by_chain[r["chain"]][1] += 1
        by_chain[r["chain"]][0] += int(r["passed"])
    print("per primitive (only subgoals entered from a good state):")
    for chain, (good, total) in sorted(by_chain.items()):
        print("   %-12s %d/%d" % (chain, good, total))

    print("\nTASK VERDICT: {}".format(success.detail))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"task": args.task, "language": spec["tasks"][args.task]["language"],
                       "rows": rows, "task_success": success.as_dict()}, handle, indent=1)
        print("wrote {}".format(args.out))
    return 0 if success.passed else 1


if __name__ == "__main__":
    sys.exit(main())
