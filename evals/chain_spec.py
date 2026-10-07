"""Turn evals/chains.yaml into a flat list of subgoals."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET_CHECKS = {"above_target": ("above_site", "above_object"),
                 "near_target": ("near_site", "near_object")}


def load(path: str = "") -> Dict[str, Any]:
    return yaml.safe_load(open(path or os.path.join(HERE, "chains.yaml")))


def _resolve_check(check: Dict[str, Any], fields: Dict[str, Any]) -> Dict[str, Any]:
    check = dict(check)
    name = check.get("fn")
    if name in TARGET_CHECKS:
        site_fn, body_fn = TARGET_CHECKS[name]
        is_site = fields.get("target_kind") == "site"
        check["fn"] = site_fn if is_site else body_fn
        check["site" if is_site else "body"] = fields["target"]
    return {k: (v.format(**fields) if isinstance(v, str) and "{" in v else v)
            for k, v in check.items()}


def expand(spec: Dict[str, Any], task_id: int) -> List[Dict[str, Any]]:
    """The full ordered subgoal list for one task."""
    chains, task = spec["chains"], spec["tasks"][task_id]
    out: List[Dict[str, Any]] = []
    for number, step in enumerate(task["steps"], start=1):
        fields = {k: v for k, v in step.items() if k != "chain"}
        for chain_name in step["chain"]:
            if chain_name not in chains:
                raise KeyError("task {} names chain {!r}, which is not defined"
                               .format(task_id, chain_name))
            for template in chains[chain_name]:
                entry = {
                    # Two steps of a task can run the same chain against the same target --
                    # task 8 puts two pots on one burner -- so the step number keeps the names
                    # distinct and keeps the printed order readable.
                    "name": "{}.{}".format(number, template["name"].format(**fields)),
                    "chain": chain_name,
                    "step": number,
                    "subgoal": template["subgoal"].format(**fields).strip(),
                    "criterion": template.get("criterion", "").format(**fields).strip(),
                    "max_cycles": int(template.get("max_cycles", 6)),
                    "check": _resolve_check(template["check"], fields),
                }
                if template.get("body"):
                    # What the truth probe measures against when the check itself names no
                    # body -- a grasp is scored by the gripper, but the run is only readable
                    # with the millimetres beside it.
                    entry["body"] = template["body"].format(**fields)
                out.append(entry)
    return out

