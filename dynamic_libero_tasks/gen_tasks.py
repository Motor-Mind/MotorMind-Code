"""Write the task BDDLs (+ their motion sidecars) in tasks/. Run once; outputs are checked in.

    python dynamic_libero_tasks/gen_tasks.py

Each task is one row below. The target is listed first; every other moving object is a
distractor, and none of them shares the target's spoken name.
"""
from pathlib import Path

import numpy as np
import yaml

OUT = Path(__file__).parent / "tasks"

#: how the language names each object category
NAMES = {
    "red_coffee_mug": "red mug", "porcelain_mug": "white mug",
    "white_yellow_mug": "yellow and white mug",
    "akita_black_bowl": "black bowl", "red_akita_black_bowl": "red bowl",
    "white_bowl": "white bowl", "yellow_bowl": "yellow bowl",
    "glazed_rim_porcelain_ramekin": "white ramekin", "red_ramekin": "red ramekin",
    "alphabet_soup": "alphabet soup", "tomato_sauce": "tomato sauce", "ketchup": "ketchup",
    "green_ketchup": "green ketchup", "blue_ketchup": "blue ketchup", "bbq_sauce": "bbq sauce",
    "salad_dressing": "salad dressing", "red_salad_dressing": "red salad dressing",
    "cream_cheese": "cream cheese", "red_cream_cheese": "red cream cheese",
    "butter": "butter", "green_butter": "green butter",
    "chocolate_pudding": "chocolate pudding", "green_chocolate_pudding": "green chocolate pudding",
    "cookies": "cookies", "yellow_cookies": "yellow cookies", "popcorn": "popcorn",
    "macaroni_and_cheese": "macaroni and cheese", "milk": "milk", "yellow_milk": "yellow milk",
    "orange_juice": "orange juice", "red_orange_juice": "red orange juice",
    "basket": "basket", "red_basket": "red basket", "wooden_tray": "wooden tray",
    "plate": "plate", "yellow_plate": "yellow plate",
}
#: receivers: how the goal is stated, and the preposition the language uses
CONTAINERS = {"basket", "red_basket", "wooden_tray"}          # In <x>_contain_region
PLATES = {"plate", "yellow_plate"}                            # On <x>

BELT_X = -0.10           # 0.56 m in front of the robot base (x = -0.66)
BELT_HALF = 0.45         # the belt runs y = -0.45 .. +0.45
SLOTS = (0.10, 0.25, 0.40, 0.55, 0.70)   # object spacing, metres downstream of the belt's start
CAROUSEL_C = (-0.06, 0.02)

# (target, distractors, receiver, speed m/s, +1 = belt runs toward +y, target's slot index)
CONVEYOR = [
    ("red_coffee_mug", ["porcelain_mug", "white_yellow_mug"], "basket", 0.015, +1, 1),
    ("alphabet_soup", ["tomato_sauce", "ketchup", "cream_cheese", "milk"], "basket", 0.025, -1, 1),
    ("ketchup", ["bbq_sauce", "salad_dressing", "orange_juice"], "red_basket", 0.02, +1, 2),
    ("cream_cheese", ["butter", "chocolate_pudding", "cookies"], "wooden_tray", 0.02, -1, 0),
    ("tomato_sauce", ["alphabet_soup", "macaroni_and_cheese", "popcorn"], "basket", 0.03, +1, 0),
    ("green_ketchup", ["ketchup", "blue_ketchup"], "basket", 0.015, -1, 2),
    ("white_yellow_mug", ["red_coffee_mug", "porcelain_mug"], "plate", 0.015, +1, 0),
    ("green_butter", ["butter", "cream_cheese", "red_cream_cheese"], "red_basket", 0.02, +1, 1),
    ("red_orange_juice", ["milk", "orange_juice", "yellow_milk"], "basket", 0.0175, -1, 1),
    ("green_chocolate_pudding", ["chocolate_pudding", "cookies", "yellow_cookies", "butter"],
     "wooden_tray", 0.025, +1, 0),
]

# (target, distractors, receiver, omega rad/s (+ = counter-clockwise), ring radius m)
CAROUSEL = [
    ("yellow_bowl", ["akita_black_bowl", "white_bowl", "red_akita_black_bowl"], "plate", 0.157, 0.15),
    ("white_bowl", ["yellow_bowl", "red_akita_black_bowl", "akita_black_bowl"], "yellow_plate", -0.157, 0.15),
    ("red_coffee_mug", ["porcelain_mug", "white_yellow_mug"], "plate", 0.20, 0.15),
    ("alphabet_soup", ["tomato_sauce", "bbq_sauce", "ketchup", "macaroni_and_cheese"], "basket", 0.157, 0.16),
    ("cream_cheese", ["butter", "cookies", "chocolate_pudding"], "red_basket", -0.20, 0.15),
    ("glazed_rim_porcelain_ramekin", ["red_ramekin", "akita_black_bowl", "white_bowl"], "plate", 0.12, 0.15),
    ("yellow_milk", ["milk", "orange_juice", "red_orange_juice"], "basket", -0.12, 0.15),
    ("akita_black_bowl", ["red_akita_black_bowl", "white_bowl", "yellow_bowl"], "yellow_plate", 0.25, 0.15),
    ("popcorn", ["macaroni_and_cheese", "cookies", "yellow_cookies", "alphabet_soup"], "wooden_tray", 0.157, 0.16),
    ("red_salad_dressing", ["salad_dressing", "ketchup", "bbq_sauce"], "red_basket", -0.157, 0.15),
]


def region(name, x, y, h=0.01):
    return f"""      ({name}
          (:target main_table)
          (:ranges (
              ({x - h:.4f} {y - h:.4f} {x + h:.4f} {y + h:.4f})
            )
          )
          (:yaw_rotation (
              (0.0 0.0)
            )
          )
      )"""


def receiver_xy(receiver, side):
    """Near the robot, off to one side; a tray is bigger, so it sits further out."""
    return (-0.33, side * 0.36) if receiver == "wooden_tray" else (-0.30, side * 0.32)


def write(stem, language, objects, target, receiver, motion):
    """objects: [(instance, category, x, y)] -- each placed ON its own main_table region."""
    in_goal = receiver in CONTAINERS
    goal = f"(In {target} {receiver}_contain_region)" if in_goal else f"(On {target} {receiver})"
    regions = "\n".join(region(f"{inst}_region", x, y) for inst, _, x, y in objects)
    if in_goal:                                    # an In goal names the receiver's region
        regions += f"\n      (contain_region\n          (:target {receiver})\n      )"
    decl = "\n".join(f"    {inst} - {cat}" for inst, cat, _, _ in objects)
    init = "\n".join(f"    (On {inst} main_table_{inst}_region)" for inst, *_ in objects)
    bddl = f"""(define (problem LIBERO_Dynamic_Tabletop_Manipulation)
  (:domain robosuite)
  (:language {language})
    (:regions
{regions}
    )

  (:fixtures
    main_table - table
  )

  (:objects
{decl}
  )

  (:obj_of_interest
    {target}
    {receiver}
  )

  (:init
{init}
  )

  (:goal
    (And {goal})
  )

)
"""
    (OUT / f"{stem}.bddl").write_text(bddl)
    (OUT / f"{stem}.motion.yaml").write_text(yaml.safe_dump(motion, sort_keys=False))


def sentence(kind, target, receiver):
    where = "conveyor belt" if kind == "conveyor" else "rotating platform"
    prep = "in" if receiver in CONTAINERS else "on"
    return f"pick up the {NAMES[target]} from the {where} and place it {prep} the {NAMES[receiver]}"


def stem(kind, i, target, receiver):
    return f"{kind}_{i:02d}_{NAMES[target].replace(' ', '_')}_to_{receiver}"


def conveyor(i, target, distractors, receiver, speed, sign, slot):
    cats = list(distractors)
    cats.insert(slot, target)
    start = -sign * BELT_HALF                     # upstream end, in y
    objects = [(f"{c}_1", c, BELT_X, start + sign * SLOTS[k]) for k, c in enumerate(cats)]
    rx, ry = receiver_xy(receiver, -sign)         # beside the upstream half of the belt
    objects.append((f"{receiver}_1", receiver, rx, ry))
    write(stem("conveyor", i, target, receiver), sentence("conveyor", target, receiver),
          objects, f"{target}_1", f"{receiver}_1",
          {"type": "linear", "start": [BELT_X, start], "end": [BELT_X, -start],
           "speed": speed, "loop": False, "width": 0.14,
           "moving": [f"{c}_1" for c in cats]})


def carousel(i, target, distractors, receiver, omega, ring):
    cats = [target] + list(distractors)
    phase = np.deg2rad(37 * i)                     # a different starting angle per task
    angles = phase + 2 * np.pi * np.arange(len(cats)) / len(cats)
    objects = [(f"{c}_1", c, CAROUSEL_C[0] + ring * np.cos(a), CAROUSEL_C[1] + ring * np.sin(a))
               for c, a in zip(cats, angles)]
    rx, ry = receiver_xy(receiver, 1 if i % 2 == 0 else -1)
    objects.append((f"{receiver}_1", receiver, rx, ry))
    write(stem("carousel", i, target, receiver), sentence("carousel", target, receiver),
          objects, f"{target}_1", f"{receiver}_1",
          {"type": "circular", "center": list(CAROUSEL_C), "omega": omega, "radius": 0.25,
           "moving": [f"{c}_1" for c in cats]})


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    for i, row in enumerate(CONVEYOR):
        conveyor(i, *row)
    for i, row in enumerate(CAROUSEL):
        carousel(i, *row)
    print(len(list(OUT.glob("*.bddl"))), "tasks written to", OUT)
