"""Tile the per-task videos into one grid video per kind:  python -m dynamic_libero_tasks.grid"""
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

VIDEOS = Path(__file__).parent / "videos"
TILE = 256


def tile(path: Path, n: int):
    frames = imageio.mimread(path, memtest=False)
    frames += [frames[-1]] * (n - len(frames))                     # hold the last frame
    label = path.stem.removeprefix("demo_")
    out = []
    for f in frames[:n]:
        f = cv2.resize(np.ascontiguousarray(f[:, : f.shape[0]]), (TILE, TILE))   # agentview half
        cv2.putText(f, label[:34], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        out.append(f)
    return out


def grid(prefix: str, name: str, cols: int = 5):
    paths = sorted(VIDEOS.glob(prefix + "*.mp4"))
    n = max(len(imageio.mimread(p, memtest=False)) for p in paths)
    tiles = [tile(p, n) for p in paths]
    rows = [np.concatenate(tiles[i:i + cols], axis=2) for i in range(0, len(tiles), cols)]
    video = np.concatenate(rows, axis=1)
    imageio.mimsave(VIDEOS / "grids" / name, list(video), fps=20, macro_block_size=8)


if __name__ == "__main__":
    (VIDEOS / "grids").mkdir(exist_ok=True)
    for kind in ("conveyor", "carousel"):
        grid(kind, f"{kind}_idle.mp4")
        grid(f"demo_{kind}", f"{kind}_oracle.mp4")
