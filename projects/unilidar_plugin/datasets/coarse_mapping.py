"""Single source of truth for the coarse-class (unified) label mapping.

The eight coarse classes are:
    0 Other / 1 Driveable Ground / 2 Structure / 3 Vehicle
    4 Nature / 5 Living Being / 6 Objects / 7 Other Ground
Coarse class 0 does not enter the mIoU (evaluation applies valid_mask = gt != 0
and discards those points), so the mIoU is the mean over the remaining seven.

Every component reads the mapping from this module;
`tools/check_coarse_mapping.py` verifies that every copy agrees with it.
"""

COARSE_NAMES = {0: 'Other', 1: 'Driveable Ground', 2: 'Structure', 3: 'Vehicle',
                4: 'Nature', 5: 'Living Being', 6: 'Objects', 7: 'Other Ground'}

# --- SemanticKITTI: raw -> coarse -------------------------------------------
# The label word is read as uint32 and taken modulo 2**16 to drop the instance
# identifier, then looked up directly in this table.
SEMANTICKITTI_TO_CL = {
    0: 0, 1: 0,                                  # unlabeled / outlier
    52: 0, 99: 0,                                # other-structure / other-object
    10: 3, 13: 3, 16: 3, 18: 3, 20: 3,           # car/bus/on-rails/truck/other-vehicle
    11: 6, 15: 6,                                # bicycle / motorcycle
    30: 5, 31: 5, 32: 5,                         # person/bicyclist/motorcyclist
    40: 1, 44: 1, 60: 1,                         # road/parking/lane-marking
    48: 7, 49: 7, 72: 7,                         # sidewalk/other-ground/terrain
    50: 2, 51: 2,                                # building/fence
    70: 4, 71: 4,                                # vegetation/trunk
    80: 6, 81: 6,                                # pole/traffic-sign
    252: 3, 256: 3, 257: 3, 258: 3, 259: 3,      # moving-* vehicles
    253: 5, 254: 5, 255: 5,                      # moving-bicyclist/person/motorcyclist
}

# --- nuScenes: two stages. raw(0-31) -> 16 classes -> coarse -----------------
NUSCENES_LEARNING_MAP = {
    0: 0, 29: 0, 31: 0,                          # noise / static.other / vehicle.ego
    1: 7, 5: 7, 7: 7, 8: 7,                      # animal/personal_mobility/stroller/wheelchair
    2: 7, 3: 7, 4: 7, 6: 7,                      # pedestrian adult/child/worker/police
    9: 1, 12: 8,                                 # barrier / trafficcone
    10: 8, 11: 8, 13: 8,                         # debris/pushable_pullable/bicycle_rack
    14: 2, 21: 6,                                # bicycle / motorcycle (riders included; not separable in the official annotation)
    15: 3, 16: 3, 17: 4, 18: 5, 22: 9, 23: 10,   # bus.bendy/bus.rigid/car/construction/trailer/truck
    19: 4, 20: 4,                                # emergency.ambulance / emergency.police
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16,
}
NUSCENES_TO_CL = {                               # 16 classes -> coarse
    0: 0, 1: 6, 2: 6, 3: 3, 4: 3, 5: 3, 6: 6, 7: 5, 8: 6,
    9: 3, 10: 3, 11: 1, 12: 7, 13: 7, 14: 7, 15: 2, 16: 4,
}
NUSCENES_RAW_TO_CL = {r: NUSCENES_TO_CL[c] for r, c in NUSCENES_LEARNING_MAP.items()}

# --- Waymo: raw -> coarse ----------------------------------------------------
# The label file holds two int32 columns; the second one is looked up here.
WAYMO_TO_CL = {
    0: 0,
    1: 3, 2: 3, 3: 3, 4: 3,                      # CAR/TRUCK/BUS/OTHER_VEHICLE
    5: 5, 6: 5, 7: 5,                            # MOTORCYCLIST/BICYCLIST/PEDESTRIAN
    8: 6, 9: 6, 10: 6, 11: 6, 12: 6, 13: 6,      # SIGN/TRAFFIC_LIGHT/POLE/CONE/BICYCLE/MOTORCYCLE
    14: 2,                                       # BUILDING
    15: 4, 16: 4,                                # VEGETATION/TREE_TRUNK
    18: 1, 19: 1, 20: 1,                         # ROAD/LANE_MARKER/OTHER_GROUND
    17: 7, 21: 7, 22: 7,                         # CURB/WALKABLE/SIDEWALK
}

# Inherent differences in annotation granularity that no mapping can remove,
# disclosed as such in the paper.
INHERENT_GAPS = [
    "nuScenes has no separate pole/sign/traffic-light classes: the official "
    "static.manmade definition already covers them together with buildings and "
    "walls and is not separable -> they fall under Structure in nuScenes, "
    "whereas SemanticKITTI and Waymo annotate them separately and map them to "
    "Objects",
    "nuScenes riders are not separable from their vehicle: the official "
    "annotation instruction is 'If there is a rider and/or passenger, include "
    "them in the box' -> in nuScenes an occupied two-wheeler and its rider both "
    "fall under Objects, whereas SemanticKITTI and Waymo annotate the rider "
    "separately and map it to Living Being",
    "Waymo has no fence class: its Structure holds buildings only",
]
