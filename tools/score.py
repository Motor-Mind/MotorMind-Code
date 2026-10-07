#!/usr/bin/env python
"""Score libero_pro sweeps side by side and bucket the failures of the last one.

usage: tools/score.py NAME=GLOB [NAME=GLOB ...]   e.g. tools/score.py c3='qwen-c3-*-0929' c4='qwen-c4-0929-*'
Globs are relative to $STORM_RUNS (default <repo>/evals/runs). The last run named is the one whose
failures are bucketed, and "passed in X, failed in last" flip sets are listed against the others.
"""
import collections, glob, json, os, sys

RUNS = os.environ.get("STORM_RUNS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals", "runs")


def cells(pattern):
    out = {}
    for f in glob.glob(os.path.join(RUNS, pattern, "task*_seed*.json")):
        j = json.load(open(f))
        if j.get("status") in ("crashed", "killed") or "task_success" not in j:
            continue
        m = j.get("mission", {})
        subs = m.get("subgoals", [])
        out[(j["task"], j["seed"])] = dict(
            run=os.path.basename(os.path.dirname(f)), variant=j.get("variant", "?"),
            passed=bool(j["task_success"]["passed"]), status=m.get("status", "?"),
            replans=max(len(m.get("plans", [])) - 1, 0), min=round(m.get("elapsed_s", 0) / 60, 1),
            died=subs[-1]["name"] if subs else "", lang=j.get("language", ""))
    return out


def main(specs):
    runs = {n: cells(p) for n, p in (s.split("=", 1) for s in specs)}
    names = list(runs)
    variants = sorted({c["variant"] for r in runs.values() for c in r.values()})
    print("| variant | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for v in variants + ["TOTAL"]:
        row = [sum(c["passed"] for c in r.values() if v in ("TOTAL", c["variant"])) for r in runs.values()]
        print(f"| {v} | " + " | ".join(map(str, row)) + " |")
    last = runs[names[-1]]
    fails = {k: c for k, c in last.items() if not c["passed"]}
    print(f"\n{names[-1]} failures: {len(fails)}  by mission status:",
          dict(collections.Counter(c["status"] for c in fails.values()).most_common()))
    for n in names[:-1]:
        flips = sorted(k for k, c in fails.items() if runs[n].get(k, {}).get("passed"))
        print(f"passed in {n}, failed in {names[-1]}: {len(flips)} {flips}")
    print("\ntask seed variant status replans min died_at")
    for k, c in sorted(fails.items()):
        print(k[0], k[1], c["variant"], c["status"], c["replans"], c["min"], c["died"])


if __name__ == "__main__":
    main(sys.argv[1:])
