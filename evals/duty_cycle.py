"""How a mission spent its wall clock: moving, or waiting for a model.

    python evals/duty_cycle.py evals/runs/<dir> [evals/runs/<other dir> ...]
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evals.where_the_clock_went import one_from  # noqa: E402


def numbers(data, path=""):
    """One row: the whole of :func:`where_the_clock_went.one`, plus the two ratios this table
    prints that it does not."""
    row = dict(one_from(data, path))
    row.update({"wall_s": row["wall"], "motion_s": row["motion"],
                "idle": 1.0 - (row["motion"] / row["wall"] if row["wall"] else 0.0),
                "calls": row["proposes"] + row["locates"] + row["supervises"],
                "of": int(str(row["conjuncts"]).split("/")[1] or 0),
                "conjuncts": int(str(row["conjuncts"]).split("/")[0] or 0),
                "propose_calls": row["proposes"], "locate_calls": row["locates"],
                "supervise_calls": row["supervises"]})
    return row


def main(paths):
    print("| run | status | conj | cycles | steps | wall s | motion s | idle | calls | "
          "calls/motion s | propose s | locate s | superv s |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    totals = []
    for path in paths:
        for name in sorted(glob.glob(os.path.join(path, "*.json"))):
            try:
                row = numbers(json.load(open(name)), name)
            except Exception as exc:                                  # a stub, or a crash
                print("| {} | unreadable: {} |".format(os.path.basename(name), exc))
                continue
            totals.append(row)
            print("| {}/{} | {} | {}/{} | {} | {} | {:.0f} | {:.0f} | {:.0%} | {} | {:.2f} "
                  "| {:.0f} | {:.0f} | {:.0f} |"
                  .format(os.path.basename(path.rstrip("/")),
                          os.path.basename(name)[:-5], row["status"], row["conjuncts"],
                          row["of"], row["cycles"], row["steps"], row["wall_s"],
                          row["motion_s"], row["idle"], row["calls"],
                          row["calls_per_motion_s"], row["propose_s"], row["locate_s"],
                          row["supervise_s"]))
    if len(totals) > 1:
        wall = sum(r["wall_s"] for r in totals) / len(totals)
        motion = sum(r["motion_s"] for r in totals) / len(totals)
        print("\n{} runs: mean wall {:.0f} s, mean motion {:.0f} s, mean idle {:.0%}, "
              "mean calls {:.0f}, {} of {} conjuncts, {} passed"
              .format(len(totals), wall, motion, 1.0 - (motion / wall if wall else 0.0),
                      sum(r["calls"] for r in totals) / len(totals),
                      sum(r["conjuncts"] for r in totals), sum(r["of"] for r in totals),
                      sum(1 for r in totals if r["passed"])))
        timed = [r for r in totals if r["propose_s"] or r["locate_s"] or r["supervise_s"]]
        if timed:
            print("of which, per mission: propose {:.0f} s over {:.0f} calls, locate {:.0f} s "
                  "over {:.0f}, supervise {:.0f} s over {:.0f} -- {:.0%} of the wall clock is "
                  "model time"
                  .format(*[sum(r[k] for r in timed) / len(timed)
                            for k in ("propose_s", "propose_calls", "locate_s",
                                      "locate_calls", "supervise_s", "supervise_calls")],
                          sum(r["propose_s"] + r["locate_s"] + r["supervise_s"]
                              for r in timed) / max(1.0, sum(r["wall_s"] for r in timed))))


if __name__ == "__main__":
    main(sys.argv[1:] or ["evals/runs/round14-regress"])
