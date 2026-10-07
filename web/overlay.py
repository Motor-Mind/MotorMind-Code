"""Draw the frames onto a live camera image, so you can see what "forward" means."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # the overlay is optional; without cv2 the raw frame is served
    cv2 = None

# RGB, matching the page's palette.
X_COLOUR = (232, 96, 88)
Y_COLOUR = (120, 208, 132)
Z_COLOUR = (110, 168, 255)
TOOL_COLOUR = (250, 196, 90)
DIM = (150, 160, 175)


def _projector(K, cam2base, width, height):
    K = np.asarray(K, dtype=float)
    base2cam = np.linalg.inv(np.asarray(cam2base, dtype=float))

    def project(point):
        q = base2cam @ np.array([point[0], point[1], point[2], 1.0])
        if q[2] <= 1e-3:                      # behind the lens
            return None
        uv = K @ (q[:3] / q[2])
        if not (-4 * width < uv[0] < 4 * width and -4 * height < uv[1] < 4 * height):
            return None                       # absurdly far outside: drawing it is noise
        return int(round(float(uv[0]))), int(round(float(uv[1])))

    return project


def draw(rgb: np.ndarray, intrinsic, cam2base, pose=None,
         control_offset_m: float = 0.0) -> np.ndarray:
    """Return a copy of ``rgb`` with the frames drawn on it."""
    if cv2 is None or rgb is None or intrinsic is None or cam2base is None:
        return rgb
    image = np.ascontiguousarray(rgb.copy())
    height, width = image.shape[:2]
    project = _projector(intrinsic, cam2base, width, height)
    scale = width / 256.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    def line(a, b, colour, thickness=1):
        pa, pb = project(a), project(b)
        if pa and pb:
            cv2.line(image, pa, pb, colour, thickness, cv2.LINE_AA)
            return pb
        return None

    # --- the base frame, at the arm's origin
    for axis, colour, name in (((0.15, 0, 0), X_COLOUR, "x"),
                               ((0, 0.15, 0), Y_COLOUR, "y"),
                               ((0, 0, 0.15), Z_COLOUR, "z")):
        tip = line((0, 0, 0), axis, colour, 2)
        if tip:
            cv2.putText(image, name, (tip[0] + 3, tip[1] - 3), font, 0.32 * scale, colour,
                        1, cv2.LINE_AA)
    origin = project((0, 0, 0))
    if origin:
        cv2.circle(image, origin, max(2, int(3 * scale)), DIM, 1, cv2.LINE_AA)
        cv2.putText(image, "base", (origin[0] + 5, origin[1] + int(11 * scale)), font,
                    0.3 * scale, DIM, 1, cv2.LINE_AA)

    if pose is not None:
        R = np.asarray(pose.rotation, dtype=float)
        pose_p = np.asarray(pose.position_m, dtype=float)
        control = pose_p + R @ np.array([0.0, 0.0, control_offset_m])

        if control_offset_m > 1e-6:           # the stalk from the reported pose to the tool
            line(pose_p, control, DIM, 1)
        for column, colour in ((0, X_COLOUR), (1, Y_COLOUR), (2, Z_COLOUR)):
            line(control, control + R[:, column] * 0.06, colour, 1)
        marker = project(control)
        if marker:
            radius = max(3, int(5 * scale))
            cv2.circle(image, marker, radius, TOOL_COLOUR, 1, cv2.LINE_AA)
            cv2.circle(image, marker, 1, TOOL_COLOUR, -1, cv2.LINE_AA)

    return image
