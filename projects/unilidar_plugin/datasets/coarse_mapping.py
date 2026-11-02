"""粗类映射的唯一真源（single source of truth）。

8 个粗类：
    0 Other / 1 Driveable Ground / 2 Structure / 3 Vehicle
    4 Nature / 5 Living Being / 6 Objects / 7 Other Ground
粗类 0 不参与 mIoU（评测时 valid_mask = gt != 0 直接丢弃），mIoU = 其余 7 类求和除以 7。

背景（2026-09-22）：此前同一张映射表在全仓有 38 份副本（配置、诊断脚本、分发包各一套），
且 VoxelClassMapping 用 np.zeros 建查表——**漏改的键会静默变成粗类 0、不报任何错**。
本模块把三张表收敛到一处；`tools/check_coarse_mapping.py` 负责校验所有副本与本模块一致。

若要修改映射：改本文件，然后跑 tools/check_coarse_mapping.py 找出所有不一致的副本。
"""

COARSE_NAMES = {0: 'Other', 1: 'Driveable Ground', 2: 'Structure', 3: 'Vehicle',
                4: 'Nature', 5: 'Living Being', 6: 'Objects', 7: 'Other Ground'}

# --- SemanticKITTI: raw -> coarse -------------------------------------------
SEMANTICKITTI_TO_CL = {
    0: 0, 1: 0,                                  # unlabeled / outlier
    # [2026-09-22 Eason 裁定「必须改」] 52/99 改自 2/6 -> 0。
    # 依据一（官方口径）：SemanticKITTI 官方 learning_map 自己就把这两个
    #   兜底类判给 unlabeled（semantic-kitti.yaml:121 与 :135），官方定义为
    #   「点太少或类内差异过大」。粗类表此前把它们复活，与细类基准不一致。
    # 依据二（实测规模，S1 直读 .label 全量统计，train 19130 帧 23.5 亿点）：
    #   raw 99 占 Objects 训练信号 71.14%（val 75.47%），raw 52 占 Structure
    #   1.15%（val 1.84%）。即 Objects 类近四分之三的监督信号来自官方认定
    #   不可用的杂项叶子。
    # 依据三（后果，S2 同权重同映射配对混淆矩阵）：class 6 成为按排除法定义
    #   的汇，Living Being 在源域就有 28.02% 被判进 Objects、跨域 63.73%，
    #   选择性比任何其他类高 4–6 倍。
    # 注意：这是标签体系变更，既有 SK 数字全部作废，须重训重测。
    52: 0, 99: 0,                                # other-structure / other-object
    10: 3, 13: 3, 16: 3, 18: 3, 20: 3,           # car/bus/on-rails/truck/other-vehicle
    11: 6, 15: 6,                                # bicycle / motorcycle   [15 改自 3]
    30: 5, 31: 5, 32: 5,                         # person/bicyclist/motorcyclist
    40: 1, 44: 1, 60: 1,                         # road/parking/lane-marking
    48: 7, 49: 7, 72: 7,                         # sidewalk/other-ground/terrain  [72 改自 4]
    50: 2, 51: 2,                                # building/fence
    70: 4, 71: 4,                                # vegetation/trunk
    80: 6, 81: 6,                                # pole/traffic-sign
    252: 3, 256: 3, 257: 3, 258: 3, 259: 3,      # moving-* 车辆
    253: 5, 254: 5, 255: 5,                      # moving-bicyclist/person/motorcyclist
}

# --- nuScenes: 两级。raw(0-31) -> 16类 -> coarse -----------------------------
# 第一级与 projects/configs/pointocc/label_mapping/nuscenes.yaml::learning_map 同源
NUSCENES_LEARNING_MAP = {
    0: 0, 29: 0, 31: 0,                          # noise / static.other / vehicle.ego
    1: 7, 5: 7, 7: 7, 8: 7,                      # animal/personal_mobility/stroller/wheelchair  [改自 0]
    2: 7, 3: 7, 4: 7, 6: 7,                      # pedestrian adult/child/worker/police
    9: 1, 12: 8,                                 # barrier / trafficcone
    10: 8, 11: 8, 13: 8,                         # debris/pushable_pullable/bicycle_rack  [改自 0]
    14: 2, 21: 6,                                # bicycle / motorcycle（含骑车人，官方不可分离）
    15: 3, 16: 3, 17: 4, 18: 5, 22: 9, 23: 10,   # bus.bendy/bus.rigid/car/construction/trailer/truck
    19: 4, 20: 4,                                # emergency.ambulance / emergency.police  [改自 0]
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16,
}
NUSCENES_TO_CL = {                               # 16类 -> coarse
    0: 0, 1: 6, 2: 6, 3: 3, 4: 3, 5: 3, 6: 6, 7: 5, 8: 6,
    9: 3, 10: 3, 11: 1, 12: 7, 13: 7, 14: 7, 15: 2, 16: 4,   # [14 改自 4]
}
NUSCENES_RAW_TO_CL = {r: NUSCENES_TO_CL[c] for r, c in NUSCENES_LEARNING_MAP.items()}

# --- Waymo: raw -> coarse ----------------------------------------------------
WAYMO_TO_CL = {
    0: 0,
    1: 3, 2: 3, 3: 3, 4: 3,                      # CAR/TRUCK/BUS/OTHER_VEHICLE
    5: 5, 6: 5, 7: 5,                            # MOTORCYCLIST/BICYCLIST/PEDESTRIAN
    8: 6, 9: 6, 10: 6, 11: 6, 12: 6, 13: 6,      # SIGN/TRAFFIC_LIGHT/POLE/CONE/BICYCLE/MOTORCYCLE
    14: 2,                                       # BUILDING
    15: 4, 16: 4,                                # VEGETATION/TREE_TRUNK
    18: 1, 19: 1, 20: 1,                         # ROAD/LANE_MARKER/OTHER_GROUND  [20 改自 7]
    17: 7, 21: 7, 22: 7,                         # CURB/WALKABLE/SIDEWALK         [17 改自 1]
}

# 已知的、映射无法弥合的固有粒度差异（供论文如实披露）
INHERENT_GAPS = [
    "nuScenes 无独立的 pole/sign/traffic-light 细类：官方 static.manmade 定义即包含它们与建筑、墙，不可分离 -> 在 NU 属 Structure，在 SK/WA 属 Objects",
    "nuScenes 的骑车人与车不可分离：官方标注 'If there is a rider and/or passenger, include them in the box' -> 在 NU 属 Objects，在 SK/WA 属 Living Being",
    "Waymo 无 fence 细类：其 Structure 只含 building",
]
