"""Every task of a suite at the seeds asked for, as one command, summarised as one table.

    python evals/sweep_missions.py --tasks 0-9 --seeds 0,1 --jobs 10         --out evals/runs/sweep-libero10
    python evals/sweep_missions.py --suite libero_pro --tasks 0-199 --seeds 0,1         --jobs 2 --gpus 0 --round 29 --timeout-min 10

It is RESUMABLE: a mission whose JSON is already in the output directory is not run again,
so the same command after a stop picks up where it left off. Stop it with
``kill -TERM <pid>`` (or ctrl-C) -- it stops its children, and a mission killed mid-run
leaves no JSON, so the resume repeats it rather than counting it done.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evals.report_missions import report  # noqa: E402

#: How long a mission is given to stop on its own before it is killed. Its run JSON is written
#: before its videos are assembled, so this protects only the recording; a model call already
#: on the wire holds the exit for up to the client's 60 s timeout. 90 s covers that.
GRACE_S = 90.0

def ask_egl() -> Dict[int, int]:
    """CUDA GPU index -> the EGL device that renders on it, as the EGL driver reports it
    (EGL_EXT_device_query, EGL_CUDA_DEVICE_NV). The two orders differ by host: on ours CUDA 0
    is EGL 1. ``{}`` where EGL cannot be asked."""
    import ctypes
    try:
        egl = ctypes.CDLL("libEGL.so.1")
        egl.eglGetProcAddress.restype, egl.eglGetProcAddress.argtypes = ctypes.c_void_p, [
            ctypes.c_char_p]
        query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.POINTER(ctypes.c_int))(
            egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
        attrib = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_ssize_t))(
            egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
        count = ctypes.c_int()
        query(0, None, ctypes.byref(count))
        devices = (ctypes.c_void_p * count.value)()
        query(count.value, devices, ctypes.byref(count))
    except (OSError, TypeError, AttributeError):
        return {}
    table = {}
    for index in range(count.value):
        cuda = ctypes.c_ssize_t(-1)
        if attrib(devices[index], 0x323A, ctypes.byref(cuda)) and cuda.value >= 0:
            table[int(cuda.value)] = index
    return table


#: The EGL device index that renders on each CUDA GPU index, asked of this host's driver.
EGL_OF_GPU = ask_egl()


def render_env(gpu: int, table: Optional[Dict[int, int]] = None) -> Dict[str, str]:
    """The MuJoCo render env that puts a sim on CUDA GPU ``gpu``."""
    table = EGL_OF_GPU if table is None else table
    if gpu not in table:
        raise ValueError("the EGL driver reports no device for GPU {}".format(gpu))
    egl = table[gpu]
    visible = str(gpu) if egl == gpu else "{},{}".format(gpu, egl)
    return {"MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(egl), "CUDA_VISIBLE_DEVICES": visible}


def free_mb(gpu: int, run: Optional[Callable[[List[str]], str]] = None) -> int:
    """Megabytes free on that GPU, straight out of ``nvidia-smi``, or 0 if it cannot be
    asked."""
    command = ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
               "-i", str(gpu)]
    runner = run or (lambda cmd: subprocess.check_output(cmd, text=True))
    try:
        return int(runner(command).strip().splitlines()[0])
    except Exception:
        return 0


def parse_tasks(text: str) -> List[int]:
    """``"0-9"``, ``"0,3,7"`` and ``"0-2,9"`` all mean what they look like."""
    tasks: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part.lstrip("-"):
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise ValueError("{!r} counts backwards".format(part))
            tasks.extend(range(lo, hi + 1))
        else:
            tasks.append(int(part))
    if not tasks:
        raise ValueError("no tasks in {!r}".format(text))
    return sorted(dict.fromkeys(tasks))


def parse_gpus(text: str, table: Optional[Dict[int, int]] = None) -> List[int]:
    """``--gpus`` keeps the order given -- it is the round-robin order -- but drops repeats."""
    gpus = [int(p.strip()) for p in text.split(",") if p.strip()]
    if not gpus:
        raise ValueError("no GPUs in {!r}".format(text))
    unknown = [g for g in gpus if g not in (EGL_OF_GPU if table is None else table)]
    if unknown:
        raise ValueError("no EGL index known for GPU(s) {}".format(unknown))
    return list(dict.fromkeys(gpus))


def records_by_default(suite: str) -> bool:
    """Whether ``suite`` is recorded unless told otherwise.

    The LIBERO-PRO sweep is two hundred tasks nobody has watched before, so it keeps its
    video and its model calls; the regression rounds are unchanged and keep writing nothing
    but their run files, which is what makes their numbers comparable with the old ones.
    """
    from adapter.libero.pro import AGGREGATE, suite_names
    return suite == AGGREGATE or suite in suite_names()


def mission_name(task: int, seed: int) -> str:
    """What one mission's files are called. The sweep's whole resume rests on this name."""
    return "task{}_seed{}".format(task, seed)


def fresh_record_dir(out_dir: str, name: str) -> str:
    """``<out>/<name>/`` with nothing of an earlier attempt in it.

    A mission that was killed leaves no JSON, so a resume runs it again -- into the
    directory the dead attempt had been appending its calls and events to. The dead
    attempt is kept, under ``<name>.attempt<k>/``: it is the record of why that run died,
    which is the one thing a retry cannot tell you.
    """
    path = os.path.join(out_dir, name)
    if os.path.isdir(path) and os.listdir(path):
        attempt = 1
        while os.path.exists("{}.attempt{}".format(path, attempt)):
            attempt += 1
        os.rename(path, "{}.attempt{}".format(path, attempt))
    return path


#: What a sweep wrote for a mission that never wrote its own file -- a run that did not
#: happen, not a result (see :meth:`Job.finish`).
NOT_A_RESULT = ("killed", "crashed")


def already_done(path: str) -> bool:
    """Whether that mission's file is a RESULT, and so not to be run again.

    A mission the sweep killed on its timeout has a stub written for it. Over ten missions
    that is a line in a table someone reads; over four hundred it is a hole that nobody
    notices, so a resume runs those again -- and the recording of the attempt that died is
    kept beside the new one (:func:`fresh_record_dir`).
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path) as handle:
            return json.load(handle).get("status") not in NOT_A_RESULT
    except Exception:
        return True              # unreadable, but written: a resume must not loop on it


def plan_pending(tasks: List[int], seeds: List[int], out_dir: str):
    """``(still to run, already done)``: a mission with a RESULT in ``out_dir`` is done."""
    pairs = [(t, s) for t in tasks for s in seeds]
    done = [p for p in pairs
            if already_done(os.path.join(out_dir, mission_name(*p) + ".json"))]
    return [p for p in pairs if p not in done], done


def out_dir_for(out: str, round_number: Optional[int], suite: str) -> str:
    """Where the runs go: ``--out`` if given, else ``evals/runs/round<N>-<suite>``."""
    if out:
        return os.path.abspath(out)
    if round_number is None:
        raise ValueError("say where the runs go: --out <dir>, or --round <N> for "
                         "evals/runs/round<N>-<suite>")
    return os.path.join(ROOT, "evals", "runs", "round{}-{}".format(round_number, suite))


class Placer:
    """Picks the next GPU to launch on, round-robin, skipping the ones that are too full."""

    def __init__(self, gpus: List[int], min_free_mb: int, reserve_mb: int, settle_s: float,
                 clock: Callable[[], float] = time.monotonic,
                 free: Callable[[int], int] = free_mb):
        self.gpus, self.min_free_mb = gpus, min_free_mb
        self.reserve_mb, self.settle_s = reserve_mb, settle_s
        self.clock, self.free = clock, free
        self.next_index = 0
        self.reservations: List[Tuple[int, float]] = []   # (gpu, when it was booked)

    def _reserved(self, gpu: int) -> int:
        now = self.clock()
        self.reservations = [r for r in self.reservations if now - r[1] < self.settle_s]
        return self.reserve_mb * len([r for r in self.reservations if r[0] == gpu])

    def pick(self) -> Optional[Tuple[int, int]]:
        """The next GPU with room, as ``(gpu, effective free mb)``, or None if none has room."""
        for offset in range(len(self.gpus)):
            gpu = self.gpus[(self.next_index + offset) % len(self.gpus)]
            room = self.free(gpu) - self._reserved(gpu)
            if room >= self.min_free_mb:
                self.next_index = (self.next_index + offset + 1) % len(self.gpus)
                self.reservations.append((gpu, self.clock()))
                return gpu, room
        return None


class Job:
    """One ``run_mission.py`` subprocess, its GPU, its clock and its deadline."""

    def __init__(self, task: int, seed: int, out_dir: str, suite: str, passthrough: List[str],
                 timeout_s: float, gpu: int, record: bool = False):
        self.task, self.seed, self.suite, self.gpu = task, seed, suite, gpu
        self.name = mission_name(task, seed)
        self.json_path = os.path.join(out_dir, self.name + ".json")
        self.log_path = os.path.join(out_dir, self.name + ".log")
        self.record_dir = fresh_record_dir(out_dir, self.name) if record else ""
        self.timeout_s = timeout_s
        self.command = [sys.executable, os.path.join(ROOT, "evals", "run_mission.py"),
                        "--task", str(task), "--seed", str(seed), "--suite", suite,
                        "--out", self.json_path] \
            + (["--record-dir", self.record_dir] if record else []) + passthrough
        self.log = open(self.log_path, "w")
        env = dict(os.environ)
        env.update(render_env(gpu))
        self.started = time.monotonic()
        self.process = subprocess.Popen(self.command, cwd=ROOT, env=env,
                                        stdout=self.log, stderr=subprocess.STDOUT)
        self.timed_out = False

    def ask_to_stop(self) -> None:
        """SIGTERM, and no waiting: several jobs are asked together and waited for after."""
        if self.process.poll() is None:
            self.process.terminate()

    def stop(self, grace_s: float = GRACE_S) -> None:
        """SIGTERM, then SIGKILL if it is still there. A run that is recording has its
        videos in memory until it ends, so it is ASKED to stop before it is killed."""
        if self.process.poll() is not None:
            return
        self.ask_to_stop()
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return
            time.sleep(0.2)
        self.process.kill()
        self.process.wait()

    def poll(self) -> bool:
        """True once the process has ended. Stops it first if it has outstayed its timeout."""
        if self.process.poll() is None:
            if time.monotonic() - self.started > self.timeout_s:
                self.timed_out = True
                self.stop()
            else:
                return False
        self.log.close()
        return True

    def finish(self) -> str:
        """Record the run and return the line to print."""
        minutes = (time.monotonic() - self.started) / 60.0
        code = self.process.returncode
        if not os.path.exists(self.json_path):
            # "killed", not "timeout": a mission that ends on its OWN wall clock reports
            # `timeout` and is a run that finished, which is not this.
            status = "killed" if self.timed_out else "crashed"
            stub: Dict[str, Any] = {
                "task": self.task, "suite": self.suite, "seed": self.seed, "status": status,
                "exit_code": code, "gpu": self.gpu, "log": os.path.basename(self.log_path),
                "error": "{} after {:.1f} min on GPU {}; see {}".format(
                    status, minutes, self.gpu, os.path.basename(self.log_path))}
            with open(self.json_path, "w") as handle:
                json.dump(stub, handle, indent=1)
        else:
            status = "ok" if code == 0 else "not done" if code == 1 else "exit {}".format(code)
        return "{:<16} {:<9} {:.1f} min  gpu {}".format(self.name, status, minutes, self.gpu)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="0-9", help="a range or a comma list: 0-9, 0,3,7")
    parser.add_argument("--seeds", default="0,1", help="likewise, over LIBERO's init states")
    parser.add_argument("--jobs", type=int, default=10, help="missions running at once")
    parser.add_argument("--gpus", default="1,0",
                        help="CUDA indices to place sims on, round-robin, in this order")
    parser.add_argument("--min-free-mb", type=int, default=2500,
                        help="a GPU with less than this free is passed over")
    parser.add_argument("--reserve-mb", type=int, default=1200,
                        help="what a launched sim is assumed to want before nvidia-smi sees it")
    parser.add_argument("--settle-s", type=float, default=30.0,
                        help="how long that reservation is held against its GPU")
    parser.add_argument("--timeout-min", type=float, default=45.0,
                        help="a run past this is killed and recorded as a timeout")
    parser.add_argument("--record", dest="record", action="store_true", default=None,
                        help="keep every frame, model call and event of every mission, in "
                             "<out>/task<T>_seed<S>/ -- the default for the LIBERO-PRO suites")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="run files only, as the regression rounds do")
    parser.add_argument("--round", type=int, default=None,
                        help="the round this sweep is; names the output directory")
    parser.add_argument("--out", default="",
                        help="where the runs go; the default is evals/runs/round<N>-<suite>")
    args, passthrough = parser.parse_known_args()

    tasks, seeds = parse_tasks(args.tasks), parse_tasks(args.seeds)
    gpus = parse_gpus(args.gpus)
    out_dir = out_dir_for(args.out, args.round, args.suite)
    os.makedirs(out_dir, exist_ok=True)

    record = records_by_default(args.suite) if args.record is None else bool(args.record)
    pending, skipped = plan_pending(tasks, seeds, out_dir)
    print("{} runs ({} tasks x {} seeds), {} already done, {} at a time, {:.0f} min each at "
          "most".format(len(pending) + len(skipped), len(tasks), len(seeds), len(skipped),
                        args.jobs, args.timeout_min))
    print("into {}{}".format(out_dir,
                             ", recording each mission beside its run file" if record
                             else ", run files only (not recording)"))
    print("GPUs {} (EGL {}), skipping any with under {} MB free".format(
        ", ".join(str(g) for g in gpus), ", ".join(str(EGL_OF_GPU[g]) for g in gpus),
        args.min_free_mb))
    if passthrough:
        print("passing through: {}".format(" ".join(passthrough)))

    placer = Placer(gpus, args.min_free_mb, args.reserve_mb, args.settle_s)
    running: List[Job] = []
    stopping: List[bool] = []          # a list so the handler can set it without a global

    def stop_everything(number, _frame):
        # Asked to stop: launch nothing more and let the running missions finish flushing.
        # Every one is asked FIRST and waited for after, so the grace is paid once, not once
        # per job -- a recording mission spends it writing its videos out.
        stopping.append(True)
        print("\nsignal {}: stopping {} running mission(s); rerun the same command to "
              "resume".format(number, len(running)), flush=True)
        for job in running:
            job.ask_to_stop()
        for job in running:
            job.stop()

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop_everything)
    peak, on_gpu = 0, {g: 0 for g in gpus}
    waiting_since: Optional[float] = None
    while (pending and not stopping) or running:
        if pending and not stopping and len(running) < max(1, args.jobs):
            room = placer.pick()
            if room is None:
                if waiting_since is None:
                    waiting_since = time.monotonic()
                    print("no GPU with {} MB free; waiting".format(args.min_free_mb), flush=True)
            else:
                waiting_since = None
                gpu, room_mb = room
                task, seed = pending.pop(0)
                running.append(Job(task, seed, out_dir, args.suite, passthrough,
                                   args.timeout_min * 60.0, gpu, record=record))
                on_gpu[gpu] += 1
                peak = max(peak, len(running))
                print("start {:<16} gpu {} ({} MB room), {} running".format(
                    running[-1].name, gpu, room_mb, len(running)), flush=True)
        time.sleep(2.0)
        for job in [j for j in running if j.poll()]:
            running.remove(job)
            print(job.finish(), flush=True)

    print("")
    if stopping:
        left, _ = plan_pending(tasks, seeds, out_dir)
        print("stopped with {} of {} runs still to do; the same command resumes them".format(
            len(left), len(tasks) * len(seeds)))
    print("peak {} missions at once; placed {}".format(
        peak, ", ".join("{} on GPU {}".format(n, g) for g, n in sorted(on_gpu.items()))))
    print(report(out_dir))
    print("wrote {}".format(os.path.join(out_dir, "REPORT.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
