"""Everything a mission saw and everything it said, kept beside its run file.

A run JSON says what the mission decided. It does not say what the pictures looked like, and
it deliberately drops the prompts (mission.py strips them: they are thousands of characters
each). This module records the rest, so a failed mission can be watched, not only read:

* **video** -- every frame the adapter rendered, one MP4 per camera, with ``frames.csv``
  giving each frame's true wall time, subgoal and cycle, because the frames are NOT evenly
  spaced and the MP4's own timeline is a nominal :data:`FPS`.
* **calls.jsonl** -- one line per model call of every role, with the whole prompt, the whole
  reply, the tokens, the seconds, and the file names of the DOWNSCALED pictures that call
  actually sent.
* **images/** -- those pictures, named by content, so the same frame sent to three roles is
  stored once.
* **events.jsonl**, **evidence.jsonl**, **notes.json** -- the mission's event stream as the
  page shows it, the EvidenceLog rows nothing else writes out, and the summariser's notes.

**It cannot slow the mission clock.** A frame costs one ``list.append`` of the JPEG the
adapter had already encoded; a call costs one short line appended to a file. The videos are
assembled from the buffer in :meth:`MissionRecorder.finish`, after the mission has stopped
and its clock has been read.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

#: The MP4's nominal frame rate. The real frames are not evenly spaced -- ``frames.csv`` has
#: the wall time of each one, and is the record to read for timing.
FPS = 4.0
#: Where the web server records every mission and subgoal it drives, one directory each,
#: outside the repository: ~/storm-runs unless STORM_RECORD_ROOT moves it ("off" turns it off).
#: At 256 px an observation of three cameras is about 140 KB (colour and 16-bit depth), so a
#: mission comes to about 15-30 MB.
RECORD_ROOT = os.path.join(os.path.expanduser("~"), "storm-runs")


def _complain(recorder, what: str, exc: Exception) -> None:
    """A recording that cannot write is a lost record, never a lost mission."""
    import sys
    message = "recording: {} failed: {}: {}".format(what, type(exc).__name__, exc)
    try:
        recorder.problems.append(message)
    except Exception:                                              # pragma: no cover
        pass
    print(message, file=sys.stderr)


def start_recording(kind: str) -> Optional["MissionRecorder"]:
    """A recorder for one run the web server drives, wired to every model call -- or None."""
    root = os.environ.get("STORM_RECORD_ROOT", RECORD_ROOT)
    if not root or root.strip().lower() == "off":
        return None
    try:
        recorder = MissionRecorder(os.path.join(root, time.strftime("%Y%m%d-%H%M%S-") + kind),
                                   video=False)
    except OSError:
        return None
    from vlms import qwen
    qwen.RECORDER = recorder.on_call
    return recorder


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def directory_bytes(directory: str) -> int:
    """How much one recording came to, on disk."""
    total = 0
    for root, _, names in os.walk(directory):
        total += sum(_size(os.path.join(root, name)) for name in names)
    return total


class MissionRecorder:
    """One mission's recording. Not reused: one directory, one mission, one process."""

    def __init__(self, directory: str, fps: float = FPS, video: bool = True):
        self.directory = os.path.abspath(directory)
        self.fps = float(fps)
        self.want_video = bool(video)
        os.makedirs(self.directory, exist_ok=True)
        self.images_dir = os.path.join(self.directory, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._frames: List[Dict[str, Any]] = []      # camera, jpeg, w, h, t, subgoal, cycle
        self._calls = 0
        self._images = 0
        self._events = 0
        self._subgoal = ""
        self._cycle: Optional[int] = None
        self._started = time.time()
        self._calls_path = os.path.join(self.directory, "calls.jsonl")
        self._events_path = os.path.join(self.directory, "events.jsonl")
        self.problems: List[str] = []

    # ------------------------------------------------------------------ where we are

    def on_event(self, event: Dict[str, Any]) -> None:
        """The mission's own event stream: filed, and read for which subgoal and cycle the
        frames and calls that follow belong to."""
        name = (event or {}).get("event")
        with self._lock:
            if name == "subgoal_start":
                self._subgoal, self._cycle = event.get("name") or "", None
            elif name == "proposed":
                self._cycle = event.get("cycle")
            self._events += 1
        self._append(self._events_path, dict(event or {}, t=time.time()))

    # ------------------------------------------------------------------ pictures

    def on_frame(self, camera: str, jpeg: bytes, width: int = 0, height: int = 0,
                 t: Optional[float] = None) -> None:
        """One rendered camera frame, as the JPEG the adapter had already made of it."""
        if not jpeg:
            return
        with self._lock:
            # Two clocks: see ``_write_frames_csv``.
            self._frames.append({"camera": str(camera), "jpeg": bytes(jpeg),
                                 "width": int(width), "height": int(height),
                                 "wall": time.time(),
                                 "t": None if t is None else float(t),
                                 "subgoal": self._subgoal, "cycle": self._cycle})

    def on_camera(self, camera: str, frame: Dict[str, Any], tool=None, flange=None) -> None:
        """One camera of one observation as the harness measured it: its pictures (colour and
        depth, stored under their digests), the calibration they were measured with and the
        tool and flange poses at that moment (4x4, base frame), one line of ``cameras.jsonl``
        -- enough to redo any back-projection offline."""
        import base64

        import numpy as np
        try:
            row: Dict[str, Any] = {"t": time.time(), "camera": camera,
                                   "subgoal": self._subgoal, "cycle": self._cycle}
            for key in ("rgb_jpeg", "depth_png16"):
                if frame.get(key):
                    row[key] = self._save_image(base64.b64decode(frame[key]),
                                                ".jpg" if key == "rgb_jpeg" else ".png")
            row.update({key: frame.get(key) for key in (
                "intrinsic", "cam2base", "width", "height", "timestamp", "depth_scale",
                "depth_min_range_m", "depth_floor_band_share")
                if frame.get(key) is not None})
            for key, pose in (("tool", tool), ("flange", flange)):
                if pose is not None:
                    matrix = np.eye(4)
                    matrix[:3, :3], matrix[:3, 3] = pose.rotation, pose.position_m
                    row[key] = matrix.round(6).tolist()
            self._append(os.path.join(self.directory, "cameras.jsonl"), row)
        except Exception as exc:
            _complain(self, "a camera frame", exc)

    # ------------------------------------------------------------------ model calls

    def on_call(self, role: str, url: str, prompt: str, system: Optional[str],
                images: Sequence[bytes], reply: Any, latency_s: float = 0.0) -> None:
        """One request and its reply, whole. ``images`` are the DOWNSCALED JPEG bytes the
        call put on the wire -- exactly what the model saw, not what was rendered."""
        names = [self._save_image(blob) for blob in (images or [])]
        row: Dict[str, Any] = {
            "n": 0, "t": time.time(), "role": role, "url": url,
            "subgoal": self._subgoal, "cycle": self._cycle,
            "latency_s": round(float(latency_s), 3),
            "prompt": prompt or "", "system": system or "", "images": names,
        }
        for field in ("ok", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                      "salvaged_from", "error", "finish_reason"):
            value = getattr(reply, field, None)
            if value is not None:
                row[field] = value
        row["reply_text"] = getattr(reply, "text", "") or ""
        row["reply_json"] = getattr(reply, "data", None)
        with self._lock:
            self._calls += 1
            row["n"] = self._calls
        self._append(self._calls_path, row)

    def _save_image(self, blob: bytes, suffix: str = ".jpg") -> str:
        """Store one JPEG under its own digest, so a frame sent to three roles is one file."""
        if not blob:
            return ""
        name = hashlib.sha1(blob).hexdigest()[:16] + suffix
        path = os.path.join(self.images_dir, name)
        try:
            if not os.path.exists(path):
                with open(path, "wb") as handle:
                    handle.write(blob)
                with self._lock:
                    self._images += 1
        except Exception as exc:
            _complain(self, "an image", exc)
        return name

    # ------------------------------------------------------------------ the rest

    def write_evidence(self, rows: Sequence[Any]) -> None:
        """The EvidenceLog the mission kept: its own record of what happened."""
        path = os.path.join(self.directory, "evidence.jsonl")
        with open(path, "w") as handle:
            for row in rows or []:
                as_dict = row.model_dump() if hasattr(row, "model_dump") else dict(row)
                handle.write(json.dumps(as_dict, default=str) + "\n")

    def write_json(self, name: str, payload: Any) -> None:
        with open(os.path.join(self.directory, name), "w") as handle:
            json.dump(payload, handle, indent=1, default=str)

    def _append(self, path: str, row: Dict[str, Any]) -> None:
        try:
            line = json.dumps(row, default=str)
            with self._lock:
                with open(path, "a") as handle:
                    handle.write(line + "\n")
        except Exception as exc:
            _complain(self, os.path.basename(path), exc)

    # ------------------------------------------------------------------ closing up

    def finish(self) -> Dict[str, Any]:
        """Write the videos and the frame index. Safe to call twice; never raises."""
        with self._lock:
            frames, self._frames = self._frames, []
        summary: Dict[str, Any] = {"frames": len(frames), "calls": self._calls,
                                   "images": self._images, "events": self._events,
                                   "videos": [], "format": "none"}
        started = time.monotonic()
        if frames:
            self._write_frames_csv(frames)
            if self.want_video:
                summary["videos"], summary["format"] = self._write_videos(frames)
        # What the sweep's grace period has to cover before it may kill a mission.
        summary["finish_s"] = round(time.monotonic() - started, 2)
        summary["bytes"] = directory_bytes(self.directory)
        summary["problems"] = list(self.problems)
        self.write_json("recording.json", summary)
        return summary

    def _write_frames_csv(self, frames: List[Dict[str, Any]]) -> None:
        """Every frame's true wall time, in the order each camera's video plays."""
        path = os.path.join(self.directory, "frames.csv")
        index: Dict[str, int] = {}
        with open(path, "w") as handle:
            # `wall_s` is the same clock as calls.jsonl's `t`; `capture_t` is the camera's
            # own, which in the sim advances per control step and is not a wall time.
            handle.write("camera,frame,wall_s,since_start_s,capture_t,subgoal,cycle,"
                         "width,height\n")
            for frame in frames:
                camera = frame["camera"]
                index[camera] = index.get(camera, -1) + 1
                handle.write("{},{},{:.3f},{:.3f},{},{},{},{},{}\n".format(
                    camera, index[camera], frame["wall"], frame["wall"] - self._started,
                    "" if frame["t"] is None else "{:.3f}".format(frame["t"]),
                    frame["subgoal"].replace(",", " "),
                    "" if frame["cycle"] is None else frame["cycle"],
                    frame["width"], frame["height"]))

    def _write_videos(self, frames: List[Dict[str, Any]]):
        """One file per camera. MP4 if this environment can encode one, else a JPEG
        sequence -- which is a recording either way, and says which it is."""
        by_camera: Dict[str, List[Dict[str, Any]]] = {}
        for frame in frames:
            by_camera.setdefault(frame["camera"], []).append(frame)
        writer, how = _encoder()
        if writer is None:
            self.problems.append("no MP4 encoder here ({}): wrote JPEG sequences".format(how))
            return [self._write_jpegs(c, f) for c, f in sorted(by_camera.items())], "jpeg"
        made = []
        for camera, shots in sorted(by_camera.items()):
            path = os.path.join(self.directory, "{}.mp4".format(camera))
            try:
                writer(path, shots, self.fps)
                made.append(os.path.basename(path))
            except Exception as exc:
                self.problems.append("{}: {}: {}".format(camera, type(exc).__name__,
                                                         str(exc)[:120]))
                made.append(self._write_jpegs(camera, shots))
        return made, how

    def _write_jpegs(self, camera: str, shots: List[Dict[str, Any]]) -> str:
        folder = os.path.join(self.directory, camera)
        os.makedirs(folder, exist_ok=True)
        for number, frame in enumerate(shots):
            with open(os.path.join(folder, "{:05d}.jpg".format(number)), "wb") as handle:
                handle.write(frame["jpeg"])
        return camera + "/"


def _decode(jpeg: bytes):
    import numpy as np
    from PIL import Image
    import io
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))


def _encoder():
    """``(write(path, frames, fps), name)`` for the best encoder here, or ``(None, why)``."""
    try:
        import imageio.v2 as imageio          # noqa: F401
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()       # the binary, not just the import
    except Exception:
        pass
    else:
        def write_imageio(path, shots, fps):
            import imageio.v2 as imageio
            first = _decode(shots[0]["jpeg"])
            with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                    macro_block_size=1) as sink:
                for frame in shots:
                    picture = _decode(frame["jpeg"])
                    if picture.shape != first.shape:
                        continue         # a camera that changed size mid-mission: skipped
                    sink.append_data(picture)
        return write_imageio, "mp4 (imageio/libx264)"
    try:
        import cv2                            # noqa: F401
    except Exception as exc:
        return None, "neither imageio-ffmpeg nor cv2: {}".format(str(exc)[:80])

    def write_cv2(path, shots, fps):
        import cv2
        first = _decode(shots[0]["jpeg"])
        height, width = first.shape[:2]
        sink = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            for frame in shots:
                picture = _decode(frame["jpeg"])
                if picture.shape != first.shape:
                    continue
                sink.write(picture[:, :, ::-1])
        finally:
            sink.release()
    return write_cv2, "mp4 (cv2/mp4v)"
