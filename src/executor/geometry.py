"""From a pixel somebody pointed at to a point in the robot's base frame."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.controller.types import DOWN, DOWN_WITHIN_DEG, approach_axis  # noqa: F401

# What the executor's model answers in: both coordinates run 0..1000 across the picture,
# whatever its pixel size.
NORMALISED_SPAN = 1000.0


def to_pixels(point: Sequence[float], width: int, height: int) -> Tuple[float, float]:
    """A pair from a reply -> (u, v) in pixels of the image as it was sent."""
    u, v = float(point[0]), float(point[1])
    return u / NORMALISED_SPAN * float(width), v / NORMALISED_SPAN * float(height)


def box_to_px(box: Sequence[float], width: int, height: int) -> List[float]:
    """A box from a reply -> [x1, y1, x2, y2] in pixels of the image as it was sent."""
    return list(to_pixels(box[:2], width, height)) + list(to_pixels(box[2:4], width, height))


def _matrices(camera: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    K, T = camera.get("intrinsic"), camera.get("cam2base")
    if K is None or T is None:
        return None
    return np.asarray(K, dtype=float).reshape(3, 3), np.asarray(T, dtype=float).reshape(4, 4)


def usable(camera: Dict[str, Any]) -> bool:
    """Can anything in this picture be turned into a direction at all?"""
    return _matrices(camera) is not None and bool(camera.get("width")) and \
        bool(camera.get("height"))


def project(camera: Dict[str, Any], point: Sequence[float]) -> Optional[Tuple[float, float]]:
    """A base-frame point -> (u, v) in pixels."""
    pair = _matrices(camera)
    if pair is None:
        return None
    K, T = pair
    in_camera = T[:3, :3].T @ (np.asarray(point, dtype=float) - T[:3, 3])
    if in_camera[2] <= 1e-6:
        return None                       # behind the camera
    uv = K @ in_camera
    return float(uv[0] / uv[2]), float(uv[1] / uv[2])


def ray(camera: Dict[str, Any], u: float, v: float):
    """(origin, unit direction) in the base frame for the pixel (u, v)."""
    pair = _matrices(camera)
    if pair is None:
        return None, None
    K, T = pair
    direction = T[:3, :3] @ (np.linalg.inv(K) @ np.array([float(u), float(v), 1.0]))
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return None, None
    return T[:3, 3].copy(), direction / norm


def on_plane(camera: Dict[str, Any], u: float, v: float, z_m: float) -> Optional[np.ndarray]:
    """Where that ray meets the horizontal plane at ``z_m``, in the base frame."""
    origin, direction = ray(camera, u, v)
    if origin is None or abs(float(direction[2])) < 1e-3:
        return None
    t = (float(z_m) - float(origin[2])) / float(direction[2])
    return None if t <= 0 else origin + t * direction


def triangulate(camera_a: Dict[str, Any], pixel_a: Sequence[float],
                camera_b: Dict[str, Any], pixel_b: Sequence[float]):
    """The point closest to both rays, and how far apart they pass."""
    o1, d1 = ray(camera_a, *pixel_a)
    o2, d2 = ray(camera_b, *pixel_b)
    if o1 is None or o2 is None:
        return None, None
    A = np.array([[d1 @ d1, -(d1 @ d2)], [d1 @ d2, -(d2 @ d2)]])
    b = np.array([(o2 - o1) @ d1, (o2 - o1) @ d2])
    if abs(float(np.linalg.det(A))) < 1e-12:            # parallel rays meet nowhere useful
        return None, None
    t = np.linalg.solve(A, b)
    if t[0] <= 0 or t[1] <= 0:                          # the meeting is behind a camera
        return None, None
    p1, p2 = o1 + t[0] * d1, o2 + t[1] * d2
    return (p1 + p2) / 2.0, float(np.linalg.norm(p1 - p2))


# How far to push a footprint away from the camera, as a share of the object's own width on the
# table.
FOOTPRINT_PUSH = 0.25


def footprint_on_plane(camera: Dict[str, Any], box_px: Sequence[float], z_m: float,
                       push: float = FOOTPRINT_PUSH) -> Optional[np.ndarray]:
    """Where the object in ``box_px`` stands on the plane at ``z_m``."""
    x1, y1, x2, y2 = (float(v) for v in box_px[:4])
    bottom = contact_row(camera, box_px)
    point = on_plane(camera, (x1 + x2) / 2.0, bottom, z_m)
    if point is None or push <= 0:
        return point
    left, right = on_plane(camera, x1, bottom, z_m), on_plane(camera, x2, bottom, z_m)
    if left is None or right is None:
        return point
    width = float(np.linalg.norm((left - right)[:2]))
    eye = np.asarray(camera["cam2base"], dtype=float)[:3, 3]
    away = point[:2] - eye[:2]
    span = float(np.linalg.norm(away))
    if span < 1e-9:
        return point
    out = point.copy()
    out[:2] = point[:2] + away / span * width * push
    return out


# How far a camera's line of sight may lean off the vertical and still count as "looking
# straight down".
NADIR_WITHIN_DEG = 20.0


def looking_down(camera: Dict[str, Any]) -> bool:
    """Is this camera pointed very nearly straight down?"""
    pair = _matrices(camera)
    if pair is None:
        return False
    axis = pair[1][:3, :3] @ np.array([0.0, 0.0, 1.0])     # the line of sight, in base
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return False
    return float(-axis[2] / norm) >= float(np.cos(np.radians(float(NADIR_WITHIN_DEG))))


def _lowering_moves_down(camera: Dict[str, Any], v: float) -> bool:
    """Does a point imaged on row ``v`` move DOWN the picture as it is lowered? For a pinhole
    that rate is (K d)[1] - v d_z, d being the world's -z in the camera's own axes."""
    pair = _matrices(camera)
    if pair is None:
        return True
    down = -pair[1][2, :3]
    return float((pair[0] @ down)[1] - float(v) * down[2]) >= 0.0


def stands_at_top(camera: Dict[str, Any], box_px: Sequence[float]) -> bool:
    """Does the thing in ``box_px`` meet the table along the box's TOP edge? It does where the
    camera looks down past its near side (the rig's tilted views), so lowering a point moves it
    UP the picture over the whole box."""
    if looking_down(camera):
        return False                  # from straight above the box outlines the top face
    return not any(_lowering_moves_down(camera, v) for v in (box_px[1], box_px[3]))


def contact_row(camera: Dict[str, Any], box_px: Sequence[float]) -> float:
    """The picture row, in ``box_px``'s pixels, of the edge the thing stands on."""
    rows = (float(box_px[1]), float(box_px[3]))
    return min(rows) if stands_at_top(camera, box_px) else max(rows)


def box_middle(box: Sequence[float]) -> Tuple[float, float]:
    """The middle of a box ``[x1, y1, x2, y2]``, in its own units."""
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def box_aim(camera: Dict[str, Any], box: Sequence[float]) -> Tuple[float, float]:
    """The point in a box to treat as the object, in the box's own units."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    if looking_down(camera):
        return box_middle(box)
    if camera.get("width") and stands_at_top(camera, box_to_px(box, camera["width"],
                                                                 camera["height"])):
        return (x1 + x2) / 2.0, min(y1, y2)
    return (x1 + x2) / 2.0, max(y1, y2)


def box_on_plane(camera: Dict[str, Any], box_px: Sequence[float],
                 z_m: float, top_z_m: Optional[float] = None) -> Optional[np.ndarray]:
    """Where the object in ``box_px`` stands on the plane, by whichever rule this camera's
    own geometry makes right."""
    if looking_down(camera):
        corners = box_on_plane_corners(camera, box_px, z_m, top_z_m)
        return None if corners is None else corners.mean(axis=0)
    edge = footprint_on_plane(camera, box_px, z_m, push=0.0)
    if top_z_m is not None and edge is not None:
        middle = on_plane(camera, *box_middle(box_px), (float(z_m) + float(top_z_m)) / 2.0)
        # A thing standing on the table cannot have its middle NEARER the camera than its own
        # near edge, so a height that puts it there is not this thing's: measured on task 1, a
        # face read from a descent that stopped on something else came back 138 mm on an 18 mm
        # box and moved three looks 67-124 mm off.
        eye = np.asarray(camera["cam2base"], dtype=float)[:3, 3]
        if middle is not None and float(np.linalg.norm(middle[:2] - eye[:2])) \
                >= float(np.linalg.norm(edge[:2] - eye[:2])):
            return middle
    return footprint_on_plane(camera, box_px, z_m)


def box_on_plane_corners(camera: Dict[str, Any], box_px: Sequence[float], z_m: float,
                         top_z_m: Optional[float] = None) -> Optional[np.ndarray]:
    """The object's outline, as points in the base frame: from the side the ends of the box's
    bottom edge on the table; from above the footprint of an upright prism up to ``top_z_m``
    -- where the box read on the table and on the top face overlap, which leaves out the side
    walls the box also takes in -- or, with no top measured, the box on the table."""
    x1, y1, x2, y2 = (float(v) for v in box_px[:4])
    top, bottom = min(y1, y2), max(y1, y2)
    if not looking_down(camera):
        points = [on_plane(camera, u, contact_row(camera, box_px), z_m) for u in (x1, x2)]
        return None if any(p is None for p in points) else np.asarray(points, dtype=float)
    planes = [float(z_m)] + ([] if top_z_m is None else [float(top_z_m)])
    rings = [[on_plane(camera, u, v, z) for u in (x1, x2) for v in (top, bottom)] for z in planes]
    if any(p is None for ring in rings for p in ring):
        return None
    # The picture's own two directions laid on the table, along which each read is a rectangle.
    axes = np.asarray(camera["cam2base"], dtype=float)[:2, :2].T
    ends = np.asarray([[axes @ p[:2] for p in ring] for ring in rings])  # plane, corner, axis
    low, high = ends.min(axis=1).max(axis=0), ends.max(axis=1).min(axis=0)
    # Along an axis the two reads do not overlap on, the top face's is the extent there is.
    apart = low > high
    low, high = (np.where(apart, ends[-1].min(axis=0), low),
                 np.where(apart, ends[-1].max(axis=0), high))
    flat = [np.linalg.solve(axes, [u, v]) for v in (high[1], low[1]) for u in (low[0], high[0])]
    return np.asarray([[x, y, planes[-1]] for x, y in flat], dtype=float)


# --------------------------------------------------------- along the jaws, and across them


def along(vector, axis=None) -> float:
    """How far ``vector`` goes along the approach axis (straight down when none is given)."""
    return float(np.dot(np.asarray(vector, dtype=float), DOWN if axis is None else axis))


def across(offset, axis=None) -> np.ndarray:
    """An offset less its part along the approach axis: across the table, pointing down."""
    offset = np.asarray(offset, dtype=float)
    if axis is None or axis[2] == -1.0 or offset.shape[0] < 3:
        return offset[:2]
    return offset - along(offset, axis) * np.asarray(axis, dtype=float)


def to_level(pose, z_m) -> Optional[float]:
    """Metres from the tool point along the jaws to the level ``z_m``; None, tilted off it."""
    axis = approach_axis(pose)
    if pose is None or z_m is None or -axis[2] < np.cos(np.radians(DOWN_WITHIN_DEG)):
        return None
    return (float(pose.position_m[2]) - float(z_m)) / -float(axis[2])


# ---------------------------------------------------------------- how good is the answer

# One oblique view's error is what is left of the near-edge bias after :data:`FOOTPRINT_PUSH`
# has undone a quarter of it, floored at what one fixed view is worth even on something with no
# width at all: back-projected onto the commissioned table plane it lands 25-40 mm from the
# object across five scenes.
PLANE_ERROR_FLOOR_MM = 20.0
# Two rays that meet, or one that comes straight down: there is no near edge to undo, so what
# is left is the pointing itself and the calibration.
CENTRED_ERROR_MM = 10.0
# ...except for the wrist looking down from up high with no fixed view to check it.
UNCHECKED_FROM_HEIGHT_MM = 76.0
# How tall the things in LIBERO's scenes get, for the one error that scales with a height nobody
# has measured (plane_error_mm): the fallback where the robot declares no max_object_height_m.
MAX_UNMEASURED_HEIGHT_MM = 160.0


def plane_error_mm(camera: Dict[str, Any], box_px: Sequence[float], z_m: float,
                   top_z_m: Optional[float] = None, tallest_mm: Optional[float] = None):
    """How far this box's back-projection may be from the middle of the thing, and why."""
    if looking_down(camera):
        if top_z_m is None:
            # Straight down at a box that outlines the object's TOP, with nothing measured about
            # how far up that is: the point comes out pushed away from the camera by the
            # object's own height, which is exactly the class of error this returns.
            return (max(CENTRED_ERROR_MM,
                        float(tallest_mm or MAX_UNMEASURED_HEIGHT_MM)
                        * float(np.tan(np.radians(NADIR_WITHIN_DEG)))),
                    "a view looking down at the top of something whose height nothing measured")
        return CENTRED_ERROR_MM, "a view looking straight down at the middle of it"
    width = longest_across(box_on_plane_corners(camera, box_px, z_m))
    if width is None:
        return PLANE_ERROR_FLOOR_MM, "one oblique view's footprint"
    return (max(PLANE_ERROR_FLOOR_MM, FOOTPRINT_PUSH * width),
            "one oblique view's footprint, of something about {:.0f} mm across".format(width))


# How much lower in the picture another object's bottom edge has to be before it counts as
# standing NEARER the camera rather than beside.
OCCLUSION_MARGIN = 0.01


def hiding_the_base(box: Sequence[float],
                    others: Sequence[Sequence[float]]) -> Optional[Sequence[float]]:
    """The box, out of ``others``, whose silhouette the bottom edge of ``box`` falls inside."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    bottom, middle = max(y1, y2), (x1 + x2) / 2.0
    margin = OCCLUSION_MARGIN * NORMALISED_SPAN
    for other in others or []:
        ox1, oy1, ox2, oy2 = (float(v) for v in other[:4])
        if not (min(ox1, ox2) <= middle <= max(ox1, ox2)):
            continue
        if max(oy1, oy2) <= bottom + margin:
            continue
        if min(oy1, oy2) >= bottom:
            continue
        return other
    return None


# ----------------------------------------------------------- a face you could push on


def on_upright_plane(camera: Dict[str, Any], u: float, v: float,
                     through: Sequence[float]) -> Optional[np.ndarray]:
    """Where the ray through (u, v) meets the upright plane standing across the view at
    ``through``."""
    pair = _matrices(camera)
    if pair is None:
        return None
    anchor = np.asarray(through, dtype=float)
    eye = np.asarray(pair[1][:3, 3], dtype=float)
    azimuth = anchor[:2] - eye[:2]
    span = float(np.linalg.norm(azimuth))
    if span < 1e-6:
        return None                       # the camera is straight over it: no plane to meet
    azimuth = azimuth / span
    origin, direction = ray(camera, u, v)
    if origin is None:
        return None
    denominator = float(azimuth @ direction[:2])
    if abs(denominator) < 1e-6:
        return None
    t = float(azimuth @ (anchor[:2] - origin[:2])) / denominator
    return None if t <= 0 else origin + t * direction


# ------------------------------------------------------- depth, read once to commission the table


def points_in_box(camera: Dict[str, Any], box_px: Sequence[float], depth_m: np.ndarray,
                  step: int = 4) -> Optional[np.ndarray]:
    """Every depth return inside ``box_px``, in the base frame."""
    pair = _matrices(camera)
    if pair is None or depth_m is None:
        return None
    K, T = pair
    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box_px[:4])
    left, right = max(0, int(min(x1, x2))), min(width - 1, int(max(x1, x2)))
    top, bottom = max(0, int(min(y1, y2))), min(height - 1, int(max(y1, y2)))
    if right <= left or bottom <= top:
        return None
    rows, columns = np.mgrid[top:bottom + 1:step, left:right + 1:step]
    z = depth_m[rows, columns]
    good = z > 0
    if not bool(good.any()):
        return None
    z = z[good]
    x = (columns[good] - K[0, 2]) / K[0, 0] * z
    y = (rows[good] - K[1, 2]) / K[1, 1] * z
    return (T[:3, :3] @ np.stack([x, y, z], 1).T).T + T[:3, 3]


# ---------------------------------------------------------------- reading an outline
#
# A located outline is points in the base frame: four corners from straight above, the two
# ends of a bottom edge from across the table. These answer the questions the grasp rules ask
# of it.


def extent_along(outline: Optional[Sequence[Sequence[float]]],
                 axis: Optional[Sequence[float]]) -> Optional[float]:
    """How far the outline reaches along ``axis``, in millimetres."""
    if outline is None or axis is None:
        return None
    axis = np.asarray(axis, dtype=float)                  # 3-D, with the jaws tilted
    points = np.asarray(outline, dtype=float)[:, :len(axis)]
    if points.shape[0] < 2 or points.shape[1] < len(axis):
        return None
    along = points @ axis
    return float(np.max(along) - np.min(along)) * 1000.0


def longest_across(outline: Optional[Sequence[Sequence[float]]]) -> Optional[float]:
    """The longest way across the outline, whichever way it lies, in millimetres."""
    if outline is None:
        return None
    points = np.asarray(outline, dtype=float)[:, :2]
    if points.shape[0] < 2:
        return None
    gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return float(np.max(gaps)) * 1000.0


def outline_centroid(outline: Optional[Sequence[Sequence[float]]],
                     needs_corners: int = 3) -> Optional[np.ndarray]:
    """The middle of a located outline: xy, metres -- or None when it is not an outline."""
    if outline is None:
        return None
    points = np.asarray(outline, dtype=float)
    if points.ndim != 2 or points.shape[0] < needs_corners:
        return None
    return np.asarray(points[:, :2].mean(axis=0), dtype=float)


def measures_more(outline, held, from_the_wrist: bool = False) -> bool:
    """Does a new outline say more about how big the thing is than the one held? One from
    above (four corners) always; one from the side over another, the wrist's over nothing."""
    corners = 0 if held is None else len(held)
    return outline is not None and (len(outline) >= 4 or (
        corners < 4 and not (from_the_wrist and corners)))


def about_its_point(outline: Optional[Sequence[Sequence[float]]], point):
    """The outline, or None when its middle is further from the point it was located with than
    the outline is across: then it outlines some other thing."""
    middle = outline_centroid(outline, 2)
    if middle is None or point is None:
        return outline
    off_mm = float(np.linalg.norm(middle - np.asarray(point, dtype=float)[:2])) * 1000.0
    return outline if off_mm <= (longest_across(outline) or 0.0) else None


def over_the_outline(outline: Optional[Sequence[Sequence[float]]],
                     position_m: Optional[Sequence[float]], slack_m: float) -> bool:
    """Is that position over the outline, within ``slack_m`` along both of its directions?"""
    if outline is None or position_m is None:
        return False
    points = np.asarray(outline, dtype=float)[:, :2]
    if points.shape[0] < 2:
        return False
    here = np.asarray(position_m, dtype=float)[:2]
    middle = points.mean(axis=0)
    flat = points - middle
    _, _, rows = np.linalg.svd(flat, full_matrices=False)
    for axis in np.asarray(rows, dtype=float)[:2]:
        along = flat @ axis
        offset = float((here - middle) @ axis)
        if not (float(np.min(along)) - slack_m <= offset <= float(np.max(along)) + slack_m):
            return False
    return True


def depth_metres(camera: Dict[str, Any]):
    """A camera's depth image in metres, or None. The same decode the clearance reading uses."""
    blob = (camera or {}).get("depth_png16")
    if not blob:
        return None
    try:
        import base64

        import cv2
    except Exception:                                              # pragma: no cover
        return None
    try:
        raw = cv2.imdecode(np.frombuffer(base64.b64decode(blob), np.uint8),
                           cv2.IMREAD_UNCHANGED)
    except Exception:                                              # pragma: no cover
        return None
    if raw is None:
        return None
    return raw.astype(np.float32) * float((camera or {}).get("depth_scale", 0.001))



