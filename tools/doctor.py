#!/usr/bin/env python
"""Is this machine ready to run storm? Checks, in order, what a sweep or a mission needs, and says
what to do about each thing that is not.

    python tools/doctor.py              # the simulator: packages, LIBERO assets, GPUs, servers
    python tools/doctor.py --no-gpu     # skip the GPU checks (a laptop driving remote servers)

Exits 1 if anything a run needs is missing.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PACKAGES = ("numpy", "yaml", "cv2", "mujoco", "robosuite", "libero", "fastapi")
failures = []


def say(ok: bool, what: str, fix: str = "") -> None:
    print("{} {}{}".format("ok  " if ok else "FAIL", what, "" if ok else "\n     -> " + fix))
    if not ok:
        failures.append(what)


def packages() -> None:
    missing = [name for name in PACKAGES if importlib.util.find_spec(name) is None]
    say(not missing, "python packages ({})".format(", ".join(PACKAGES)),
        "missing {}: conda env create -f environment.yml, then tools/setup_third_party.sh"
        .format(", ".join(missing)))


def libero() -> None:
    from adapter.libero.adapter import SIM_LIBERO
    absent = [d for d in ("bddl_files", "init_files")
              if not os.path.isdir(os.path.join(SIM_LIBERO, d))]
    say(not absent, "LIBERO scenes in sim/libero", "missing {}".format(", ".join(absent)))
    spec = importlib.util.find_spec("libero")
    package = next(iter(spec.submodule_search_locations)) if spec else ""
    say(os.path.isdir(os.path.join(SIM_LIBERO, "assets"))
        or os.path.isdir(os.path.join(package, "libero", "assets")),
        "LIBERO meshes and textures (sim/libero/assets, else the LIBERO install's)",
        "install LIBERO: tools/setup_third_party.sh")
    if os.environ.get("LIBERO_CONFIG_PATH"):
        print("note LIBERO_CONFIG_PATH={} overrides sim/libero; tools/sweep.sh unsets it"
              .format(os.environ["LIBERO_CONFIG_PATH"]))


def gpus() -> None:
    from evals.sweep_missions import EGL_OF_GPU
    try:
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
                                        "--format=csv,noheader,nounits"], text=True, timeout=20)
    except Exception as exc:
        say(False, "nvidia-smi", "no NVIDIA driver answered: {}".format(exc))
        return
    for row in rows.strip().splitlines():
        index, free, total = (int(v) for v in row.split(","))
        egl = EGL_OF_GPU.get(index)
        print("     GPU {}: {:>6} of {} MB free, EGL device {}".format(
            index, free, total, "none" if egl is None else egl))
    say(bool(EGL_OF_GPU), "EGL reports a device for each GPU (offscreen MuJoCo rendering)",
        "libEGL.so.1 with EGL_EXT_device_query is needed: install the NVIDIA EGL driver libraries")


def servers() -> None:
    from vlms.endpoints import ROLES, urls_for
    for role in ROLES:
        for url in urls_for(role):
            try:
                with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as reply:
                    names = [m.get("id") for m in json.load(reply).get("data", [])]
                say(True, "{:8} {} serves {}".format(role, url, ", ".join(names)))
            except Exception as exc:
                say(False, "{:8} {}".format(role, url),
                    "not answering ({}): start it (tools/serve_qwen.sh) or point the role "
                    "elsewhere (configs/servers.yaml or QWEN_{}_URL)".format(exc, role.upper()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-gpu", action="store_true", help="skip the GPU and EGL checks")
    args = parser.parse_args()
    packages()
    libero()
    if not args.no_gpu:
        gpus()
    servers()
    print("\n{}".format("ready" if not failures else "{} problem(s)".format(len(failures))))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
