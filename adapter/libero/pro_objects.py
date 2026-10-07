"""LIBERO-PRO's extra object types, ported onto stock LIBERO's object base classes."""
from __future__ import annotations

from libero.libero.envs.base_object import register_object, register_visual_change_object
from libero.libero.envs.objects.articulated_objects import ArticulatedObject
from libero.libero.envs.objects.google_scanned_objects import GoogleScannedObject
from libero.libero.envs.objects.hope_objects import HopeBaseObject
from libero.libero.envs.objects.turbosquid_objects import TurbosquidObjects

import numpy as np

_UPRIGHT = {"x": (np.pi / 2, np.pi / 2), "z": (np.pi / 2, np.pi / 2)}
_TIPPED = (-np.pi / 2, -np.pi / 2)

# class name -> (base, attributes set after the base __init__). The asset name is the class
# name in snake_case, which is also the key register_object files it under.
_SIMPLE = {
    "YellowBowl": (GoogleScannedObject, {}),
    "BlackBowl": (GoogleScannedObject, {}),
    "RedAkitaBlackBowl": (GoogleScannedObject, {}),
    "BiggerAkitaBlackBowl": (GoogleScannedObject, {}),
    "YellowPlate": (GoogleScannedObject, {}),
    "RedBasket": (GoogleScannedObject, {}),
    "RedRamekin": (GoogleScannedObject, {}),
    "BiggerAlphabetSoup": (HopeBaseObject, {"rotation_axis": "z"}),
    "RedAlphabetSoup": (HopeBaseObject, {"rotation_axis": "z"}),
    "GreenBbqSauce": (HopeBaseObject, {}),
    "GreenButter": (HopeBaseObject, {"rotation": (0.0, 0.0), "rotation_axis": "x"}),
    "GreenChocolatePudding": (HopeBaseObject, {"rotation": (0.0, 0.0), "rotation_axis": "x"}),
    "YellowCookies": (HopeBaseObject, {}),
    "RedCreamCheese": (HopeBaseObject, {"rotation": (0.0, 0.0), "rotation_axis": "x"}),
    "GreenKetchup": (HopeBaseObject, {"rotation": _UPRIGHT, "rotation_axis": None}),
    "RedOrangeJuice": (HopeBaseObject, {"rotation": _UPRIGHT}),
    "RedSaladDressing": (HopeBaseObject, {"rotation": _UPRIGHT, "rotation_axis": None}),
    "YellowTomatoSauce": (HopeBaseObject, {"rotation_axis": "z"}),
    "BlueKetchup": (HopeBaseObject, {"rotation_axis": "z"}),
    "BiggerMilk": (HopeBaseObject, {"rotation_axis": "z"}),
    "YellowMilk": (HopeBaseObject, {"rotation_axis": "z"}),
    "BrownRack": (TurbosquidObjects, {}),
    "WineRackStand": (TurbosquidObjects, {}),
    "WhiteBottle": (TurbosquidObjects, {}),
    "YellowMokaPot": (TurbosquidObjects, {}),
    "YellowDeskCaddy": (TurbosquidObjects, {}),
    "WhitePorcelainMug": (TurbosquidObjects, {"rotation": _TIPPED}),
    "RedYellowBook": (TurbosquidObjects, {"rotation": _TIPPED}),
}


def _snake(class_name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in class_name).lstrip("_")


def _copy(value):
    """A fresh container per instance, as the hand-written classes built one each time."""
    return {k: tuple(v) for k, v in value.items()} if isinstance(value, dict) else value


def _make(class_name: str, base: type, attrs: dict) -> type:
    asset = _snake(class_name)
    if base is TurbosquidObjects:
        def __init__(self, name=asset, obj_name=asset,
                     joints=[dict(type="free", damping="0.0005")]):
            base.__init__(self, name, obj_name, joints)
            for key, value in attrs.items():
                setattr(self, key, _copy(value))
    else:
        def __init__(self, name=asset, obj_name=asset):
            base.__init__(self, name, obj_name)
            for key, value in attrs.items():
                setattr(self, key, _copy(value))
    __init__.__qualname__ = class_name + ".__init__"
    return register_object(type(class_name, (base,), {
        "__init__": __init__, "__module__": __name__, "__qualname__": class_name}))


for _name, (_base, _attrs) in _SIMPLE.items():
    globals()[_name] = _make(_name, _base, _attrs)
del _name, _base, _attrs


@register_object
class YellowCabinet(ArticulatedObject):
    def __init__(
        self,
        name="yellow_cabinet",
        obj_name="yellow_cabinet",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.object_properties["articulation"]["default_open_ranges"] = [-0.16, -0.14]
        self.object_properties["articulation"]["default_close_ranges"] = [0.0, 0.005]

    def is_open(self, qpos):
        return bool(qpos < max(self.object_properties["articulation"]["default_open_ranges"]))

    def is_close(self, qpos):
        return bool(qpos > min(self.object_properties["articulation"]["default_close_ranges"]))


@register_object
@register_visual_change_object
class YellowStove(ArticulatedObject):
    def __init__(
        self,
        name="yellow_stove",
        obj_name="yellow_stove",
        joints=[dict(type="free", damping="0.0005")],
    ):
        super().__init__(name, obj_name, joints)
        self.rotation = (0, 0)
        self.rotation_axis = "y"

        tracking_sites_dict = {}
        tracking_sites_dict["burner"] = (self.naming_prefix + "burner", False)
        self.object_properties["vis_site_names"].update(tracking_sites_dict)
        self.object_properties["articulation"]["default_turnon_ranges"] = [0.5, 2.1]
        self.object_properties["articulation"]["default_turnoff_ranges"] = [-0.005, 0.0]

    def turn_on(self, qpos):
        if qpos >= min(self.object_properties["articulation"]["default_turnon_ranges"]):
            # TODO: Set visualization sites to be true
            self.object_properties["vis_site_names"]["burner"] = (
                self.naming_prefix + "burner",
                True,
            )
            return True
        else:
            self.object_properties["vis_site_names"]["burner"] = (
                self.naming_prefix + "burner",
                False,
            )
            return False

    def turn_off(self, qpos):
        if qpos < max(self.object_properties["articulation"]["default_turnoff_ranges"]):
            self.object_properties["vis_site_names"]["burner"] = (
                self.naming_prefix + "burner",
                False,
            )
            return True
        else:
            self.object_properties["vis_site_names"]["burner"] = (
                self.naming_prefix + "burner",
                True,
            )
            return False
