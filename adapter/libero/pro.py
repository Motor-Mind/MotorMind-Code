"""Register LIBERO-PRO's perturbed task suites as first-class LIBERO benchmarks."""
from __future__ import annotations

import re
from pathlib import Path

#: The LIBERO-PRO dimensions that are real BDDL edits; see the module docstring for why
#: all seven numbered categories -- occlusion included -- are absent.
PRO_VARIANTS = ("lan", "object", "swap", "task")

#: The base suites LIBERO-PRO perturbs (it does not touch ``libero_90``).
BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")

#: The whole benchmark under ONE name, so a sweep can say ``--suite libero_pro --tasks 0-199``
#: instead of twenty commands. Each base suite is followed by its four perturbations, so
#: global index ``i`` is ``ALL_SUITES[i // 10]`` task ``i % 10``: 0-49 spatial, 50-99 object,
#: 100-149 goal, 150-199 libero_10, and within each block 0-9 is the ORIGINAL task and 10-19,
#: 20-29, 30-39, 40-49 its lan, object, swap and task perturbation.
AGGREGATE = "libero_pro"

#: How many tasks each suite here ships. LIBERO-PRO keeps the base suite's ten.
TASKS_PER_SUITE = 10

_LANGUAGE = re.compile(r"\(:language\s+(.*?)\)", re.S)

_registered = False


def language_from_bddl(path: str | Path) -> str | None:
    """The ``(:language ...)`` line of a BDDL file, whitespace-collapsed."""
    text = Path(path).read_text(errors="replace").replace("\r\n", "\n")
    m = _LANGUAGE.search(text)
    return " ".join(m.group(1).split()) if m else None


def suite_names() -> list[str]:
    return [f"{b}_{v}" for b in BASE_SUITES for v in PRO_VARIANTS]


def all_suites() -> list[str]:
    """The twenty suites of :data:`AGGREGATE`, in its index order: each base, then its four
    perturbations."""
    return [s for b in BASE_SUITES for s in (b, *(f"{b}_{v}" for v in PRO_VARIANTS))]


def split_index(index: int) -> tuple[str, int]:
    """Global task ``index`` of :data:`AGGREGATE` as the ``(suite, task id)`` it stands for."""
    suites = all_suites()
    index = int(index)
    if not 0 <= index < len(suites) * TASKS_PER_SUITE:
        raise ValueError("{} has tasks 0-{}, not {}".format(
            AGGREGATE, len(suites) * TASKS_PER_SUITE - 1, index))
    return suites[index // TASKS_PER_SUITE], index % TASKS_PER_SUITE


def register() -> list[str]:
    """Add every available LIBERO-PRO suite to LIBERO's benchmark registry."""
    global _registered
    if _registered:
        return [s for s in suite_names() if s in _mapping()]

    from libero.libero import get_libero_path
    from libero.libero.benchmark import (Benchmark, Task, libero_task_map,
                                         register_benchmark, task_maps)
    from libero.libero.envs.base_object import OBJECTS_DICT

    from . import pro_objects               # noqa: F401  -- importing registers the types

    # LIBERO-PRO's own alias fix: three `*_object` BDDLs name a mug type that even their
    # registry lacks, and their objects/__init__.py falls back to the plain mug.
    for bad, good in (("white_white_porcelain_mug", "white_porcelain_mug"),
                      ("white_yellow_porcelain_mug", "yellow_porcelain_mug"),
                      ("white_red_porcelain_mug", "red_porcelain_mug")):
        if bad not in OBJECTS_DICT:
            if good in OBJECTS_DICT:
                OBJECTS_DICT[bad] = OBJECTS_DICT[good]
            elif "porcelain_mug" in OBJECTS_DICT:
                OBJECTS_DICT[bad] = OBJECTS_DICT["porcelain_mug"]

    bddl_root = Path(get_libero_path("bddl_files"))
    init_root = Path(get_libero_path("init_states"))
    added: list[str] = []

    for base in BASE_SUITES:
        for variant in PRO_VARIANTS:
            suite = f"{base}_{variant}"
            d = bddl_root / suite
            if not d.is_dir():
                continue
            # base order, filtered to what this variant actually ships: index k stays
            # the same underlying task across every variant of a suite
            names = [n for n in libero_task_map[base] if (d / f"{n}.bddl").is_file()]
            if not names:
                continue
            libero_task_map[suite] = names
            task_maps[suite] = {
                n: Task(
                    name=n,
                    language=language_from_bddl(d / f"{n}.bddl") or n.replace("_", " "),
                    problem="Libero",
                    problem_folder=suite,
                    bddl_file=f"{n}.bddl",
                    init_states_file=f"{n}.pruned_init",
                )
                for n in names
            }
            cls = type(suite.upper(), (Benchmark,), {
                "__init__": _make_init(suite),
                "__doc__": f"LIBERO-PRO {variant!r} perturbation of {base}.",
            })
            register_benchmark(cls)
            added.append(suite)
            if not (init_root / suite).is_dir():
                # The adapter DOES read these: `_states_for` torch.loads the `.pruned_init`
                # file and `--seed k` starts from its state k, exactly as on a base suite. A
                # suite without them is seed 0 (the plain reset) only.
                print("[pro] {} ships no initial states: only seed 0 will run".format(suite))
    _aggregate(added)
    _registered = True
    return added


def _aggregate(added: list[str]) -> bool:
    """Register :data:`AGGREGATE`: all twenty suites end to end, indexed 0-199.

    It is deliberately NOT in what :func:`register` returns -- that list is the perturbation
    suites it found, and this is a view over them.
    """
    from libero.libero.benchmark import (Benchmark, libero_task_map,
                                         register_benchmark, task_maps)

    suites = all_suites()
    if any(s not in task_maps or s not in libero_task_map for s in suites):
        return False                      # a partial vendoring has no 200-task benchmark
    names: list[str] = []
    tasks = {}
    for suite in suites:
        for name in libero_task_map[suite]:
            # The ten filenames repeat across the variants, so the key carries its suite;
            # every Task keeps its own problem_folder, which is what actually routes.
            key = f"{suite}/{name}"
            names.append(key)
            tasks[key] = task_maps[suite][name]
    libero_task_map[AGGREGATE] = names
    task_maps[AGGREGATE] = tasks

    def _make_benchmark(self) -> None:
        # NOT Benchmark's: that one permutes by `task_orders[0]`, which is ten entries long
        # and would silently cut two hundred tasks down to ten.
        self.tasks = list(task_maps[self.name].values())
        self.n_tasks = len(self.tasks)

    cls = type(AGGREGATE.upper(), (Benchmark,), {
        "__init__": _make_init(AGGREGATE),
        "_make_benchmark": _make_benchmark,
        "__doc__": "Every LIBERO-PRO suite end to end: {} tasks, index i = {} task i % {}."
                   .format(len(names), "all_suites()[i // 10]", TASKS_PER_SUITE),
    })
    register_benchmark(cls)
    return True


def _make_init(suite: str):
    def __init__(self, task_order_index: int = 0) -> None:
        from libero.libero.benchmark import Benchmark
        Benchmark.__init__(self, task_order_index=task_order_index)
        self.name = suite
        self._make_benchmark()
    return __init__


def _mapping() -> dict:
    from libero.libero.benchmark import BENCHMARK_MAPPING
    return BENCHMARK_MAPPING
