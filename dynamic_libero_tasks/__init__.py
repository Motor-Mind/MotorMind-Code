"""Dynamic LIBERO tasks: objects ride a conveyor belt or a carousel until grasped."""
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"

#: The benchmark name. Its BDDL folder is sim/libero/bddl_files/libero_dynamic, a symlink to
#: TASKS_DIR; task i is the i-th BDDL in sorted order: 0-9 carousel, 10-19 conveyor.
SUITE = "libero_dynamic"

_registered = False


def register() -> None:
    """Point LIBERO at this repo's assets, register the dynamic problem class (and the
    LIBERO-PRO object types some tasks use, e.g. yellow_bowl) and the ``libero_dynamic``
    benchmark. Call before making an env."""
    global _registered
    if _registered:
        return
    from adapter.libero.adapter import use_repo_libero
    use_repo_libero()
    from adapter.libero import pro_objects  # noqa: F401  -- registers extra object types
    from adapter.libero.pro import language_from_bddl
    from libero.libero.benchmark import (Benchmark, Task, libero_task_map,
                                         register_benchmark, task_maps)

    from . import env  # noqa: F401  -- registers Libero_Dynamic_Tabletop_Manipulation

    names = [p.stem for p in task_files()]
    libero_task_map[SUITE] = names
    task_maps[SUITE] = {p.stem: Task(name=p.stem, language=language_from_bddl(p),
                                     problem="Libero", problem_folder=SUITE,
                                     bddl_file=p.name, init_states_file=p.stem + ".pruned_init")
                        for p in task_files()}

    def __init__(self, task_order_index: int = 0) -> None:
        Benchmark.__init__(self, task_order_index=task_order_index)
        self.name = SUITE
        self._make_benchmark()

    def _make_benchmark(self) -> None:
        # NOT Benchmark's: that one permutes by `task_orders[0]`, which is ten entries long
        self.tasks = list(task_maps[SUITE].values())
        self.n_tasks = len(self.tasks)

    register_benchmark(type(SUITE.upper(), (Benchmark,), {
        "__init__": __init__, "_make_benchmark": _make_benchmark,
        "__doc__": "Dynamic LIBERO: conveyor and carousel tasks (dynamic_libero_tasks/)."}))
    _registered = True


def task_files():
    return sorted(TASKS_DIR.glob("*.bddl"))
