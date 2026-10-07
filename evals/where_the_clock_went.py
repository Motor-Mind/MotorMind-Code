"""Where a mission's wall clock went, from the run JSON and nothing else.

    python evals/where_the_clock_went.py "evals/runs/<dir>/*.json"
"""
import glob, json, sys, os

def one(path):
    """Where one run file's wall clock went."""
    return one_from(json.load(open(path)), path)


def one_from(d, path=""):
    """The same reading, off an already-loaded mission payload."""
    m = d.get("mission") or {}
    wall = float(m.get("elapsed_s") or 0.0)
    plan_s = sum(float(p.get("elapsed_s") or 0.0) for p in (m.get("plans") or []))
    plan_n = len(m.get("plans") or [])
    motion = propose = locate = supervise = 0.0
    verify = note = 0.0
    cyc_wall = sub_wall = 0.0
    cycles = proposes = locates = supervises = steps = 0
    observations = 0
    untimed = untimed_locate_n = alerts = 0
    untimed_propose_s = timed_locate_s = timed_in_cycle = 0.0
    timed_locate_n = 0
    for sub in m.get("subgoals") or []:
        sub_wall += float(sub.get("elapsed_s") or 0.0)
        v = sub.get("verdict") or {}
        verify += float(v.get("elapsed_s") or 0.0)
        n = sub.get("note") or {}
        note += float(n.get("elapsed_s") or 0.0)
        # The monitor's own seconds are NOT here.
        alerts += len(sub.get("alerts") or [])
        for c in sub.get("cycles") or []:
            cycles += 1
            t = c.get("timing") or {}
            pr = c.get("proposal") or {}
            lo = (c.get("located") or {}).get("located") or {}
            if t:
                cyc_wall += float(t.get("wall_s") or 0.0)
                motion += float(t.get("motion_s") or 0.0)
                propose += float(t.get("propose_s") or 0.0)
                locate += float(t.get("locate_s") or 0.0)
                supervise += float(t.get("supervise_s") or 0.0)
                proposes += int(t.get("propose_calls") or 0)
                locates += int(t.get("locate_calls") or 0)
                supervises += int(t.get("supervise_calls") or 0)
                timed_locate_s += float(t.get("locate_s") or 0.0)
                timed_locate_n += int(t.get("locate_calls") or 0)
                timed_in_cycle += (float(t.get("motion_s") or 0.0)
                                   + float(t.get("propose_s") or 0.0)
                                   + float(t.get("locate_s") or 0.0)
                                   + float(t.get("supervise_s") or 0.0))
            else:
                # A cycle whose every propose retry was refused ends before cycle.timing is
                # written, so its seconds are invisible to the timing block.
                untimed += 1
                untimed_propose_s += float(pr.get("elapsed_s") or 0.0)
                propose += float(pr.get("elapsed_s") or 0.0)
                proposes += len(pr.get("attempts") or [])
                untimed_locate_n += int(lo.get("calls") or 0)
                locates += int(lo.get("calls") or 0)
            ran = len(c.get("steps") or [])
            steps += ran
            b = c.get("batch") or {}
            observations += 1 + max(0, ran - 1) + (1 if b.get("stopped_at") is not None
                                                   or c.get("stopped_by") else 0)
    final = m.get("final") or {}
    verify += float((final.get("verdict") or {}).get("elapsed_s") or 0.0) \
        if isinstance(final.get("verdict"), dict) else 0.0
    per_locate = timed_locate_s / timed_locate_n if timed_locate_n else 0.0
    locate += per_locate * untimed_locate_n
    model = plan_s + propose + locate + supervise + verify + note
    goal = (d.get("ground_truth_end") or {}).get("goal") or []
    gave_up = sum(1 for s in (m.get("subgoals") or [])
                  if ((s.get("verdict") or {}).get("verdict") or {}).get("status") == "abort")
    return {
        "name": os.path.basename(path).replace(".json", ""),
        "passed": bool((d.get("task_success") or {}).get("passed")),
        "conjuncts": "{}/{}".format(sum(1 for c in goal if c.get("true")), len(goal)),
        "status": m.get("status") or "",
        "wall": wall, "motion": motion,
        "plan_s": plan_s, "propose_s": propose, "locate_s": locate,
        "supervise_s": supervise, "verify_s": verify, "note_s": note, "alerts": alerts,
        "model_s": model, "other_s": wall - model - motion,
        "cycles": cycles, "proposes": proposes, "locates": locates, "supervises": supervises,
        "steps": steps, "observations": observations,
        "calls_per_motion_s": (proposes + locates + supervises) / motion if motion else 0.0,
        "replans": max(0, plan_n - 1), "gave_up": gave_up,
        # where the "other" sits
        "in_cycles_other": cyc_wall - timed_in_cycle,
        "in_subgoal_outside_cycles": (sub_wall - cyc_wall - verify - note
                                      - untimed_propose_s - per_locate * untimed_locate_n),
        "outside_subgoals": wall - sub_wall - plan_s,
        "untimed": untimed, "untimed_propose_s": untimed_propose_s,
        "untimed_locate_s": per_locate * untimed_locate_n,
    }


def main(patterns):
    for p in sorted(sum([glob.glob(a) for a in patterns], [])):
        try:
            r = one(p)
        except Exception as exc:
            print(p, "unreadable:", exc); continue
        print("== {name}  {status}  predicate {passed}  conjuncts {conjuncts}".format(**r))
        print("   wall {wall:7.1f}s   motion {motion:6.1f}s ({mp:.0f}%)   model {model_s:6.1f}s "
              "({mm:.0f}%)   sim/other {other_s:6.1f}s ({mo:.0f}%)".format(
                  mp=100*r["motion"]/r["wall"], mm=100*r["model_s"]/r["wall"],
                  mo=100*r["other_s"]/r["wall"], **r))
        print("   model by role: plan {plan_s:.1f}  propose {propose_s:.1f}  locate {locate_s:.1f}"
              "  supervise {supervise_s:.1f}  verify {verify_s:.1f}  note {note_s:.1f}"
              "  monitor NOT TIMED ({alerts} alerts)".format(**r))
        print("   cycles {cycles}  proposes {proposes}  locates {locates}  supervises {supervises}"
              "  steps {steps}  calls/motion-s {calls_per_motion_s:.2f}  replans {replans}"
              "  give-ups {gave_up}".format(**r))
        print("   cycles with NO timing block (every propose retry refused): {untimed}"
              "  -- {untimed_propose_s:.1f}s of propose and ~{untimed_locate_s:.1f}s of locate the "
              "timing block never saw".format(**r))
        print("   other breaks down: in-cycle {in_cycles_other:6.1f}s over ~{observations} "
              "observations ({per:.2f}s each)   in-subgoal-outside-cycles "
              "{in_subgoal_outside_cycles:6.1f}s   outside-subgoals {outside_subgoals:6.1f}s".format(
                  per=r["in_cycles_other"]/r["observations"] if r["observations"] else 0.0, **r))


if __name__ == "__main__":
    main(sys.argv[1:])
