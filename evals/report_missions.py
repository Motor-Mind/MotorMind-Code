"""One markdown table for a directory of ``run_mission.py`` output.

    python evals/report_missions.py evals/runs/sweep-libero10
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

COLUMNS = ("task", "suite", "seed", "status", "predicate", "subgoals", "replans",
           "monitor stops", "cycles", "min", "died at", "task sentence")


def _named(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Which suite this run's task came from, and what the task said.

    Under an aggregate suite -- `libero_pro` is two hundred tasks over twenty suites -- the
    index alone names nothing, so the file carries the suite the task was actually in; older
    files without it fall back to ``suite``.
    """
    suite = payload.get("variant") or payload.get("suite") or "-"
    return {"suite": suite, "task sentence": (payload.get("language") or "-").strip()}


def _verdict_status(subgoal: Dict[str, Any]) -> str:
    """What the verifier said about one subgoal: its status, or ``invalid`` if the reply was
    refused."""
    call = subgoal.get("verdict") or {}
    said = call.get("verdict") or {}
    return said.get("status") or ("invalid" if not call.get("ok", True) else "none")


def _goal_at_end(payload: Dict[str, Any]) -> str:
    """Each conjunct of the BDDL goal and whether it was true when the run stopped."""
    row = payload.get("ground_truth_end")
    if not isinstance(row, dict):
        # a run that died before the final reading still has its last subgoal's
        subgoals = (payload.get("mission") or {}).get("subgoals") or []
        seen = [s.get("ground_truth") for s in subgoals
                if isinstance(s, dict) and isinstance(s.get("ground_truth"), dict)]
        row = seen[-1] if seen else {}
    goal = row.get("goal")
    if not isinstance(goal, list) or not goal:
        return "-"
    return ", ".join("`{}({})`={}".format(entry.get("predicate", "?"),
                                          ", ".join(entry.get("args") or []),
                                          "T" if entry.get("true") else "F")
                     for entry in goal if isinstance(entry, dict))


def row_from_mission(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One table row from one loaded mission file, or None if the file is not a mission."""
    mission = payload.get("mission")
    if not isinstance(mission, dict):
        if not payload.get("status"):
            return None
        return {"task": payload.get("task", "?"), "seed": payload.get("seed", "-"),
                "status": payload["status"], "predicate": "-", "subgoals": "-",
                "replans": "-", "monitor stops": "-", "cycles": "-", "min": "-",
                "died at": "-", "_died_name": "-", "_goal": _goal_at_end(payload),
                **_named(payload)}

    subgoals = mission.get("subgoals") or []
    plans = mission.get("plans") or []
    # The final task-level check is filed as a row named "task".
    steps = [s for s in subgoals if s.get("name") != "task"]
    done = sum(1 for s in steps if _verdict_status(s) == "done")
    planned = len(((plans[0] if plans else {}).get("plan") or {}).get("subgoals") or [])
    invalid = sum(1 for p in plans if not p.get("ok", True))
    replans = max(0, len(plans) - 1)
    status = mission.get("status") or "?"
    # A run that stopped because a replan could not be funded and one that ran out of clock
    # inside a subgoal both say `timeout`, and they are different findings: the first had
    # seconds left and nothing it could buy with them. Both are counted, apart.
    ended_on = {"replan_refused_for_price": "price", "clock_ran_out": "clock"}.get(
        mission.get("ended_on") or "", "")
    if status == "timeout":
        ended_on = ended_on or ("price" if mission.get("replan_refused_for_price")
                                else "clock")
        # The seconds are in the refusal block below the table; the status is a BUCKET, and
        # a bucket with a number in it is a bucket of one.
        status = "{} ({})".format(status, ended_on)

    if status == "task_done" or not subgoals:
        died = "-"
    else:
        last = subgoals[-1]
        died = "{} ({})".format(last.get("name", "?"), _verdict_status(last))

    return {
        "task": payload.get("task", "?"),
        "seed": payload.get("seed", "-"),
        **_named(payload),
        "status": status,
        "predicate": "true" if (payload.get("task_success") or {}).get("passed") else "false",
        "subgoals": "{} / {} ({})".format(done, len(steps), planned),
        "replans": "{}{}".format(replans, ", invalid" if invalid else ""),
        "monitor stops": sum(1 for s in subgoals if s.get("stopped_by_monitor")),
        "cycles": mission.get("total_cycles", "-"),
        "min": "{:.1f}".format(float(mission.get("elapsed_s") or 0.0) / 60.0),
        "died at": died,
        # not columns: the histogram groups on one, the goal block prints the other
        "_died_name": "-" if died == "-" else (subgoals[-1].get("name") or "?"),
        "_goal": _goal_at_end(payload),
        # ...and what a run that stopped with work outstanding and no clock to pay for it
        # would have needed. A give-up for price is not a run that ran out of things to try.
        "_price": mission.get("replan_refused_for_price") or {},
    }


def _sort_key(row: Dict[str, Any]):
    def number(value):
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, 0)
    return (number(row["task"]), number(row["seed"]))


def collect(directory: str):
    """Every mission row in a directory, sorted by task then seed, and the files skipped."""
    rows, skipped = [], []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        try:
            row = row_from_mission(json.load(open(path)))
        except Exception as exc:
            skipped.append("{} ({})".format(os.path.basename(path), str(exc)[:60]))
            continue
        if row is None:
            skipped.append(os.path.basename(path))
        else:
            rows.append(row)
    return sorted(rows, key=_sort_key), skipped


def _table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = ["| " + " | ".join(COLUMNS) + " |",
             "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in COLUMNS) + " |")
    return lines


def _histogram(title: str, counts: collections.Counter) -> List[str]:
    if not counts:
        return []
    lines = ["", "**{}**".format(title), ""]
    for name, count in counts.most_common():
        lines.append("* `{}` {}".format(name, count))
    return lines


def render(rows: List[Dict[str, Any]], skipped: List[str]) -> str:
    successes = sum(1 for r in rows if r["predicate"] == "true")
    lines = _table(rows)
    lines += ["", "**{} success / {} runs** (the environment's own goal predicate).".format(
        successes, len(rows))]
    # The prefix before the first underscore is the subgoal KIND -- above_soup and
    # above_handle are one failure, not two -- which is the grouping every mission reading so
    # far has turned out to want.
    kinds = collections.Counter(r["_died_name"].split("_")[0] for r in rows
                                if r["_died_name"] != "-")
    lines += _histogram("died at, by subgoal kind", kinds)
    # The status already says which of the two endings a `timeout` was, so the histogram
    # counts them apart without being told again.
    lines += _histogram("statuses", collections.Counter(str(r["status"]) for r in rows))
    # Two hundred tasks over twenty suites: the per-suite count is the reading the whole
    # sweep was run for, and a ten-task round has one line here and loses nothing.
    per_suite = collections.Counter(str(r["suite"]) for r in rows)
    if len(per_suite) > 1:
        passed = collections.Counter(str(r["suite"]) for r in rows if r["predicate"] == "true")
        lines += ["", "**success by suite** (the environment's own goal predicate)", ""]
        lines += ["* `{}` {} / {}".format(suite, passed.get(suite, 0), count)
                  for suite, count in sorted(per_suite.items())]
    priced = [r for r in rows if r.get("_price")]
    if priced:
        lines += ["", "**replans refused for price** -- work was outstanding and the clock "
                      "could not pay for it", ""]
        lines += ["* task {} seed {}: {:.0f} s left, a replan needs {:.0f} s, and the steps "
                  "for clause(s) {} cost {:.0f} s the first time (plan call {:.0f} s){}".format(
                      r["task"], r["seed"], r["_price"].get("left_s", 0.0),
                      r["_price"].get("rail_s", 0.0),
                      ", ".join(str(c) for c in r["_price"].get("outstanding") or []) or "-",
                      r["_price"].get("steps_already_cost_s", 0.0),
                      r["_price"].get("plan_call_s", 0.0),
                      # ...and what each clause still outstanding would have cost to finish,
                      # which is what the refusal was actually weighed against.
                      "".join("\n  * clause {}: {}, needs {:.0f} s".format(
                          c.get("clause"), c.get("state", "?"), c.get("needs_s", 0.0))
                          for c in r["_price"].get("clauses") or []))
                  for r in priced]
    told = [r for r in rows if r.get("_goal", "-") != "-"]
    if told:
        lines += ["", "**goal predicates at the end**", ""]
        lines += ["* task {} seed {}: {}".format(r["task"], r["seed"], r["_goal"])
                  for r in told]
    if skipped:
        lines += ["", "Not missions, skipped: {}".format(", ".join(skipped))]
    return "\n".join(lines) + "\n"


def report(directory: str) -> str:
    """Render the directory and leave ``REPORT.md`` beside the runs. Returns the markdown."""
    rows, skipped = collect(directory)
    text = render(rows, skipped)
    with open(os.path.join(directory, "REPORT.md"), "w") as handle:
        handle.write(text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory")
    args = parser.parse_args()
    if not os.path.isdir(args.directory):
        print("no such directory: {}".format(args.directory))
        return 2
    print(report(args.directory))
    print("wrote {}".format(os.path.join(args.directory, "REPORT.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
