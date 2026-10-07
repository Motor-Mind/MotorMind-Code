"""The mission's record of what happened, written by code and never by a model."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    """One line of what actually happened, with the numbers behind it kept beside it."""

    model_config = ConfigDict(extra="forbid")

    t: float
    subgoal: str = ""                 # the Subgoal.name it belongs to ("" for plan-level rows)
    kind: Literal["cycle", "supervision", "alert", "verdict", "plan", "note"]
    text: str                         # ONE human-readable line of what happened
    # command, measured mm, finding, gripper, clearance -- whatever the event carried.
    data: Dict[str, Any] = Field(default_factory=dict)


# Rendered beside a row's line.
NUMBERS = ("cycle", "along_axis_mm", "lateral_mm", "turned_deg", "clearance_mm",
           "residual_mm")


class EvidenceLog:
    """An append-only list of :class:`Evidence`, with one in-place merge (see the docstring)."""

    def __init__(self, rows: Optional[Iterable[Evidence]] = None):
        self.rows: List[Evidence] = list(rows or [])
        self._open: Dict[str, Evidence] = {}      # the cycle row still being measured, per subgoal
        # Where the current ATTEMPT at a subgoal starts in ``rows``.
        self._since: Dict[str, int] = {}

    # ------------------------------------------------------------------ writing

    def record(self, kind: str, subgoal: str, text: str,
               data: Optional[Dict[str, Any]] = None, t: Optional[float] = None) -> Evidence:
        row = Evidence(t=float(time.time() if t is None else t), subgoal=subgoal or "",
                       kind=kind, text=text, data=dict(data or {}))
        self.rows.append(row)
        return row

    def from_executor_event(self, subgoal_name: str, event: Dict[str, Any]) -> Optional[Evidence]:
        """Turn one of ``ExecutorLoop.run``'s events into a row, or ``None`` if it carries no
        fact."""
        name = (event or {}).get("event")
        if name == "proposed":
            if event.get("cycle") == 0:
                self._since[subgoal_name or ""] = len(self.rows)   # a fresh attempt starts here
            said = event.get("proposal") or {}            # None when the proposal never parsed
            data = {"cycle": event.get("cycle"), "command": said.get("command"),
                    "assessment": said.get("assessment", ""), "done": bool(event.get("done"))}
            if event.get("error"):
                data["error"] = event["error"]
            row = self.record("cycle", subgoal_name, _cycle_text(data), data)
            self._open[subgoal_name or ""] = row
            return row
        if name == "step_end":
            row = self._open.get(subgoal_name or "")
            if row is None:
                return None                               # a step with no proposal in front of it
            steps = row.data.setdefault("steps", [])
            steps.append({k: event.get(k) for k in
                          ("label", "kind", "outcome", "message", "along_axis_mm",
                           "lateral_mm", "turned_deg", "abort_reason")})
            row.data.update(_measurements(steps))
            row.text = _cycle_text(row.data)
            return row
        if name == "reached":
            # A reach makes no direction and no distance of its own: what it leaves behind is
            # a measured gap and the label the cameras put on what they aimed at, which the
            # scene monitor reads to judge whether the tool is over the right object.
            row = self._open.get(subgoal_name or "")
            if row is None:
                return None
            row.data.update(_reach(event))
            row.text = _cycle_text(row.data)
            return row
        if name == "supervision":
            if (event.get("decision") or "") != "stop":
                return None                    # a continue authorises nothing and records nothing
            return self._stop_row(subgoal_name, event)
        if name in ("limited", "refused", "step_error"):
            said = event.get("message") or event.get("error") or ""
            return self.record("cycle", subgoal_name, "{} on cycle {}: {}".format(
                name, self._cycle_now(subgoal_name), said),
                {"event": name, "label": event.get("label", ""), "message": said})
        return None

    def record_cycles(self, subgoal_name: str, cycles: List[Dict[str, Any]]) -> None:
        """Write (or complete) one row per cycle from ``ExecutorLoop.run``'s returned cycles."""
        for cycle in cycles or []:
            said = (cycle.get("proposal") or {}).get("proposal") or {}
            data: Dict[str, Any] = {"cycle": cycle.get("index"), "command": said.get("command"),
                                    "assessment": said.get("assessment", ""),
                                    "done": bool((cycle.get("proposal") or {}).get("done"))}
            data.update(_measurements(cycle.get("steps") or []))
            data.update(_reach(cycle.get("reach") or {}))
            for key in ("stopped_by", "finding", "error"):
                if cycle.get(key):
                    data[key] = cycle[key]
            if (cycle.get("proposal") or {}).get("error") and not data.get("error"):
                data["error"] = cycle["proposal"]["error"]
            self._upsert_cycle(subgoal_name, data)
            for check in cycle.get("supervision") or []:
                if (check.get("decision") or "") == "stop":
                    self._stop_row(subgoal_name, check, cycle=cycle.get("index"))
        # The attempt is over: a later one under the same name writes its own rows.
        self._since.pop(subgoal_name or "", None)
        self._open.pop(subgoal_name or "", None)

    # ------------------------------------------------------------------ reading

    def render(self, rows: Optional[Iterable[Evidence]] = None) -> str:
        """One line per row: what it was, whose subgoal, what happened, and the numbers."""
        chosen = self.rows if rows is None else list(rows)
        return "\n".join("[{}] {}: {}{}".format(r.kind, r.subgoal or "plan", r.text,
                                                _numbers(r.data)) for r in chosen)

    def as_dict(self) -> Dict[str, Any]:
        return {"rows": [r.model_dump() for r in self.rows]}

    # ------------------------------------------------------------------ internals

    def _this_attempt(self, subgoal_name: str) -> List[Evidence]:
        """The rows of the attempt now running -- all of them if none is open (see __init__)."""
        return self.rows[self._since.get(subgoal_name or "", len(self.rows)):]

    def _cycle_now(self, subgoal_name: str) -> Any:
        row = self._open.get(subgoal_name or "")
        return None if row is None else row.data.get("cycle")

    def _stop_row(self, subgoal_name: str, check: Dict[str, Any],
                  cycle: Optional[int] = None) -> Evidence:
        index = cycle if cycle is not None else self._cycle_now(subgoal_name)
        capture = check.get("capture_time", 0.0)
        for row in self._this_attempt(subgoal_name):  # the live path may have logged it already
            if row.kind == "supervision" and row.subgoal == (subgoal_name or "") \
                    and row.data.get("cycle") == index and row.data.get("capture_time") == capture:
                return row
        return self.record("supervision", subgoal_name,
                           "the watcher STOPPED cycle {} ({}): {}".format(
                               index, check.get("finding") or "no finding",
                               _trim(check.get("because") or "no reason given", 160)),
                           {"cycle": index, "finding": check.get("finding", ""),
                            "capture_time": capture,
                            "evidence": _trim(check.get("evidence") or "", 200)})

    def _upsert_cycle(self, subgoal_name: str, data: Dict[str, Any]) -> Evidence:
        # A cycle's own row is the one carrying a "command" key; the limited/refused/step_error
        # rows share its index but are separate facts and must not be overwritten by it.
        for row in self._this_attempt(subgoal_name):
            if row.kind == "cycle" and row.subgoal == (subgoal_name or "") \
                    and row.data.get("cycle") == data["cycle"] and "command" in row.data:
                row.text, row.data = _cycle_text(data), data
                return row
        return self.record("cycle", subgoal_name, _cycle_text(data), data)


# --------------------------------------------------------------------------- one line each


def _trim(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _say_command(command: Any) -> str:
    """A command in a few words. The model writes an axis as often as a direction word."""
    actions = command if isinstance(command, list) else (command or {}).get("actions")
    said = []
    for action in actions or []:
        kind = action.get("type", "?")
        way = action.get("direction") or (
            "axis {}".format(list(action["axis"])) if action.get("axis") is not None else "")
        if kind in ("move", "push"):
            said.append("{} {} {:.0f} mm".format(kind, way,
                                                 float(action.get("distance_mm") or 0)))
        elif kind == "rotate":
            said.append("rotate {} {:.0f} deg".format(way, float(action.get("angle_deg") or 0)))
        elif kind == "gripper":
            said.append("gripper {}".format(action.get("state", "?")))
        elif kind == "wait":
            said.append("wait {}s".format(action.get("seconds", "?")))
        else:
            said.append(kind)
    return ", then ".join(said)


def _reach(reach: Dict[str, Any]) -> Dict[str, Any]:
    """The one line a reach leaves, from either path -- the live event or the finished cycle,
    which carry the same keys."""
    out: Dict[str, Any] = {}
    if reach.get("text"):
        out["reach"] = _trim(reach["text"], 300)
    if isinstance(reach.get("residual_mm"), (int, float)):
        out["residual_mm"] = reach["residual_mm"]
    return out


def _measurements(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The numbers a cycle is judged on, summed over however many steps it was chunked into."""
    out: Dict[str, Any] = {}
    along = round(sum(float(s.get("along_axis_mm") or 0.0) for s in steps), 1)
    # Sideways travel, as the adapter measured it.
    lateral = round(sum(abs(float(s.get("lateral_mm") or 0.0)) for s in steps), 1)
    turned = round(sum(float(s.get("turned_deg") or 0.0) for s in steps), 1)
    kinds = [s.get("kind") for s in steps]
    if along or "translate" in kinds:
        out["along_axis_mm"] = along
        out["lateral_mm"] = lateral
    if turned or "rotate" in kinds:
        out["turned_deg"] = turned
    last = steps[-1] if steps else {}
    if last.get("outcome"):
        out["outcome"] = last["outcome"]
    if last.get("message"):
        out["message"] = _trim(last["message"], 120)
    return out


def _cycle_text(data: Dict[str, Any]) -> str:
    """"cycle 3: move down 50 mm -> measured 48 mm" -- the line a note is asked to cite."""
    text = "cycle {}: {}".format(data.get("cycle"),
                                 "the model said the subgoal is met" if data.get("done")
                                 else (_say_command(data.get("command")) or "no motion"))
    if data.get("assessment"):
        text += " -- \"{}\"".format(_trim(data["assessment"], 140))
    measured = []
    if data.get("reach"):
        measured.append(data["reach"])
    if "along_axis_mm" in data:
        measured.append("the robot measured {:.0f} mm along the axis{}".format(
            data["along_axis_mm"],
            "" if not data.get("lateral_mm")
            else " and {:.0f} mm sideways".format(data["lateral_mm"])))
    if "turned_deg" in data:
        measured.append("the tool turned {:.0f} deg".format(data["turned_deg"]))
    if data.get("outcome") and data["outcome"] != "done":
        measured.append("the step ended " + str(data["outcome"]))
    if data.get("message"):
        measured.append("the robot reports \"{}\"".format(data["message"]))
    if measured:
        text += " -> " + "; ".join(measured)
    if data.get("stopped_by"):
        text += " [STOPPED: {}{}]".format(_trim(data["stopped_by"], 120),
                                          "" if not data.get("finding")
                                          else ", finding " + data["finding"])
    if data.get("error"):
        text += " [ERROR: {}]".format(_trim(data["error"], 140))
    return text


def _numbers(data: Dict[str, Any]) -> str:
    seen = ["{}={:g}".format(k, float(data[k])) for k in NUMBERS
            if isinstance(data.get(k), (int, float)) and not isinstance(data.get(k), bool)]
    return " ({})".format(", ".join(seen)) if seen else ""
