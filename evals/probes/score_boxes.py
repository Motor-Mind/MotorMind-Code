"""Score the planner's boxes."""
import glob, json, os, re, sys
import numpy as np

SMALL = {"the","a","an","of","with","on","in","its","it","and","to","that","this","at","by",
         "for","is","are","from","near","beside","next","which","standing","left","right",
         "front","back","side","can","box","bottle","carton"}
PARTS = ("knob", "burner", "drawer", "handle", "rim", "lid", "face", "cook")
# What word overlap cannot settle, adjudicated by reading the task language and the body list.
OVERRIDE = {"white mug with a textured surface": "porcelain_mug_1",
            "white mug with the textured surface": "porcelain_mug_1",
            # the same mug, as the 512 planner names it: the task's "white mug" against a
            # scene holding porcelain_mug_1, white_yellow_mug_1 and red_coffee_mug_1
            "white mug": "porcelain_mug_1",
            # the drawer of the only cabinet in the scene
            "open bottom drawer": "white_cabinet_1",
            "bottom drawer": "white_cabinet_1",
            "open bottom drawer of the cabinet": "white_cabinet_1",
            # LIVING_ROOM_SCENE5's goal is (On porcelain_mug_1 plate_1) and
            # (On white_yellow_mug_1 plate_2); plate_1 projects at u=79 of 1024 in the scene
            # camera and plate_2 at u=959, so the task's "left plate" IS plate_1.
            "plate with red stripes on the left": "plate_1",
            "plate with red stripes on the right": "plate_2"}
# LIVING_ROOM_SCENE5, task 4: the one scene where the words cannot settle it and the clause can.
BY_CLAUSE = {0: ("porcelain_mug_1", "plate_1"), 1: ("white_yellow_mug_1", "plate_2")}
SCENE5 = {"plate_1", "plate_2", "porcelain_mug_1", "white_yellow_mug_1"}

def project(cam, point):
    K = np.asarray(cam["intrinsic"], float); T = np.asarray(cam["cam2base"], float)
    inc = T[:3,:3].T @ (np.asarray(point, float) - T[:3,3])
    if inc[2] <= 1e-6: return None
    uv = K @ inc
    return float(uv[0]/uv[2]), float(uv[1]/uv[2])

def words(text):
    return {w for w in re.findall(r"[a-z]+", str(text).lower())
            if w not in SMALL and len(w) > 2}

def body_words(name):
    return {w for w in re.findall(r"[a-z]+", name.lower()) if w != "main" and len(w) > 2}

rows = []
for path in sorted(glob.glob(sys.argv[1] + "/t*_s*.json")):
    d = json.load(open(path))
    truth = d.get("truth") or {}
    cams = {c: v for c, v in (truth.get("cameras") or {}).items() if v.get("intrinsic")}
    objs = {k: v for k, v in (truth.get("objects") or {}).items() if "centre_mm" in v}
    px = {n: {c: project(cam, np.asarray(o["centre_mm"], float)/1000.0)
              for c, cam in cams.items()} for n, o in objs.items()}
    clause_of = {sg["name"]: sg.get("clause", -1)
                 for sg in ((d.get("plan") or {}).get("plan") or {}).get("subgoals") or []}
    seen = set()
    for row in (d.get("boxes") or {}).get("rows") or []:
        clause = clause_of.get(row["name"], -1)
        if (row["target"], clause) in seen:
            continue
        seen.add((row["target"], clause))
        head = row["target"].split(",")[0]                    # the neighbour is after it
        tw = words(head)
        key = re.sub(r"^(the|a|an)\s+", "", head.lower().strip())
        intended = None
        if SCENE5 <= set(objs) and clause in BY_CLAUSE:
            # the clause says which mug and which plate; the head says which of the two
            mug, plate = BY_CLAUSE[clause]
            intended = plate if "plate" in tw else (mug if "mug" in tw else None)
        if intended is None:
            intended = OVERRIDE.get(key)
        if intended is None:
            best = sorted(objs, key=lambda n: -len(tw & body_words(n)))
            intended = best[0] if best and (tw & body_words(best[0])) else None
        said = re.findall(r"[a-z]+", head.lower())   # the HEAD clause names the part
        part = any(w in said for w in PARTS)
        for camera, box in sorted(row["boxes"].items()):
            cam = cams.get(camera)
            if not cam: continue
            W, H = cam["width"], cam["height"]
            cx, cy = (box[0]+box[2])/2000.0*W, (box[1]+box[3])/2000.0*H
            near, best_d, inside = None, 1e9, []
            for name, per in px.items():
                uv = per.get(camera)
                if uv is None: continue
                dist = float(np.hypot(uv[0]-cx, uv[1]-cy))
                if dist < best_d: best_d, near = dist, name
                if box[0]/1000*W <= uv[0] <= box[2]/1000*W and \
                   box[1]/1000*H <= uv[1] <= box[3]/1000*H:
                    inside.append(name)
            uv = px.get(intended, {}).get(camera) if intended else None
            err = None if uv is None else round(float(np.hypot(uv[0]-cx, uv[1]-cy)), 1)
            rows.append({"file": os.path.basename(path), "task": d["task"], "seed": d["seed"],
                         "target": row["target"], "camera": camera, "w": W, "h": H,
                         "intended": intended, "nearest": near, "inside": inside,
                         "err_px": err, "part": part, "undecidable": False,
                         "right": None if intended is None else
                                  (intended in inside or near == intended)})
out = sys.argv[2] if len(sys.argv) > 2 else "scored2.json"
json.dump(rows, open(os.path.join(sys.argv[1], out), "w"), indent=1)

print("%-10s %-5s %-55s %-20s %-20s %6s %s" % (
    "file","cam","target (head clause)","intended","boxed(nearest)","err","ok"))
for r in rows:
    print("%-10s %-5s %-55s %-20s %-20s %6s %s" % (
        r["file"], r["camera"], r["target"].split(",")[0][:55], str(r["intended"])[:20],
        str(r["nearest"])[:20], "-" if r["err_px"] is None else r["err_px"],
        {True:"Y", False:"WRONG", None:"?"}[r["right"]] + (" (part)" if r["part"] else "")))

print()
for cam in ("scene","side","wrist"):
    got = [r for r in rows if r["camera"] == cam]
    if not got: continue
    judged = [r for r in got if r["right"] is not None]
    right = [r for r in judged if r["right"]]
    whole = [r for r in right if not r["part"] and r["err_px"] is not None]
    errs = sorted(r["err_px"] for r in whole)
    side = got[0]["w"]
    print("%-5s %2d boxes | right object %d/%d (%.0f%%)%s" % (
        cam, len(got), len(right), len(judged), 100*len(right)/max(1,len(judged)),
        "" if len(judged) == len(got) else "  [%d undecidable]" % (len(got)-len(judged))))
    if errs:
        print("      whole-object boxes on the right object, centre error at %d px: "
              "n=%d median %.0f mean %.0f p90 %.0f max %.0f"
              % (side, len(errs), np.median(errs), np.mean(errs),
                 np.percentile(errs, 90), max(errs)))
        if side != 1024:
            scaled = [e*1024.0/side for e in errs]
            print("      the same, scaled to 1024 px: median %.0f max %.0f"
                  % (np.median(scaled), max(scaled)))
        parts = [r["err_px"] for r in right if r["part"] and r["err_px"] is not None]
        if parts:
            print("      (%d part-targets excluded: their truth is the parent body's centre)"
                  % len(parts))
