from prettytable import PrettyTable
import numpy as np

def cm_to_ious(cm):
    mean_ious = []
    cls_num = len(cm)
    for i in range(cls_num):
        tp = cm[i, i]
        p = cm[:, i].sum()
        g = cm[i, :].sum()
        union = p + g - tp
        mean_ious.append(tp / (union + 1e-6))
    
    return mean_ious

def format_results(mean_ious, return_dic=False):
    class_map = {
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values()) + ['mean']
    class_ious = mean_ious + [sum(mean_ious) / len(mean_ious)]
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = round(cls_iou, 3)
        x.add_row([cls_name, round(cls_iou, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x
    
# def format_results_sk(mean_ious, return_dic=False):
#     class_map = {
#             0: 'unlabeled', # outlier, other-structure, other-object
#             1: 'car', # moving-car
#             2: 'bicycle',
#             3: 'motorcycle',
#             4: 'truck', # moving-truck
#             5: 'other-vehicle',  # bus, moving-bus, moving-on-rails, moving-other, other-vehicle, on-rails
#             6: 'person', # moving-person
#             7: 'bicyclist', # moving-bicyclist
#             8: 'motorcyclist', # moving-motorcyclist
#             9: 'road',  # Includes 'lane-marking' as part of 'road'
#             10: 'parking', 
#             11: 'sidewalk', 
#             12: 'other-ground',
#             13: 'building',
#             14: 'fence',
#             15: 'vegetation',
#             16: 'trunk',
#             17: 'terrain',
#             18: 'pole',
#             19: 'traffic-sign'
#             }
    
#     x = PrettyTable()
#     x.field_names = ['class', 'IoU']
#     class_names = list(class_map.values()) + ['mean']
#     class_ious = mean_ious + [sum(mean_ious) / len(mean_ious)]
#     dic = {}
    
#     for cls_name, cls_iou in zip(class_names, class_ious):
#         dic[cls_name] = round(cls_iou, 3)
#         x.add_row([cls_name, round(cls_iou, 3)])
    
#     if return_dic:
#         return x, dic 
#     else:
#         return x  



def format_SC_results(mean_ious, return_dic=False):
    class_map = {
        1: 'non-empty',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x


def format_SSC_results(mean_ious, return_dic=False):
    class_map = {
        0: 'free',
        1: 'barrier',
        2: 'bicycle',
        3: 'bus',
        4: 'car',
        5: 'construction_vehicle',
        6: 'motorcycle',
        7: 'pedestrian',
        8: 'traffic_cone',
        9: 'trailer',
        10: 'truck',
        11: 'driveable_surface',
        12: 'other_flat',
        13: 'sidewalk',
        14: 'terrain',
        15: 'manmade',
        16: 'vegetation',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    mean_ious = sum(mean_ious[1:]) / len(mean_ious[1:])
    dic['mean'] = np.round(mean_ious, 3)
    x.add_row(['mean', np.round(mean_ious, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x
    
def format_SSC_results_sk(mean_ious, return_dic=False):
    class_map = {
            0: 'unlabeled', # outlier, other-structure, other-object
            1: 'car', # moving-car
            2: 'bicycle',
            3: 'motorcycle',
            4: 'truck', # moving-truck
            5: 'other-vehicle',  # bus, moving-bus, moving-on-rails, moving-other, other-vehicle, on-rails
            6: 'person', # moving-person
            7: 'bicyclist', # moving-bicyclist
            8: 'motorcyclist', # moving-motorcyclist
            9: 'road',  # Includes 'lane-marking' as part of 'road'
            10: 'parking', 
            11: 'sidewalk', 
            12: 'other-ground',
            13: 'building',
            14: 'fence',
            15: 'vegetation',
            16: 'trunk',
            17: 'terrain',
            18: 'pole',
            19: 'traffic-sign'
            }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    mean_ious = sum(mean_ious[1:]) / len(mean_ious[1:])
    dic['mean'] = np.round(mean_ious, 3)
    x.add_row(['mean', np.round(mean_ious, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x
    
    
def format_SSCOcc_results_waymo(mean_ious, return_dic=False):
    class_map = {
        0: 'Free',
        1: 'Vehicle',
        2: 'Pedestrian',
        3: 'Sign',
        4: 'Bicyclist',
        5: 'Traffic Light',
        6: 'Pole',
        7: 'Cons. Cone',
        8: 'Bicycle',
        9: 'Motorcycle', 
        10: 'Building',
        11: 'Vegetation',
        12: 'Tree Trunk',
        13: 'Road',
        14: 'Walkable',
        15: 'General Object',
    }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    mean_ious = sum(mean_ious[1:]) / len(mean_ious[1:])
    dic['mean'] = np.round(mean_ious, 3)
    x.add_row(['mean', np.round(mean_ious, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x
    
def format_SSCSeg_results_waymo(mean_ious, return_dic=False):
    class_map = {
    0: 'TYPE_UNDEFINED',
    1: 'TYPE_CAR',
    2: 'TYPE_TRUCK',
    3: 'TYPE_BUS',
    4: 'TYPE_OTHER_VEHICLE', # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    5: 'TYPE_MOTORCYCLIST',
    6: 'TYPE_BICYCLIST',
    7: 'TYPE_PEDESTRIAN',
    8: 'TYPE_SIGN',
    9: 'TYPE_TRAFFIC_LIGHT',
    10: 'TYPE_POLE', # Lamp post, traffic sign pole etc.
    11: 'TYPE_CONSTRUCTION_CONE', # Construction cone/pole.
    12: 'TYPE_BICYCLE',
    13: 'TYPE_MOTORCYCLE',
    14: 'TYPE_BUILDING',
    15: 'TYPE_VEGETATION', # Bushes, tree branches, tall grasses, flowers etc.
    16: 'TYPE_TREE_TRUNK',
    17: 'TYPE_CURB', # Curb on the edge of roads. This does not include road boundaries if there’s no curb.
    18: 'TYPE_ROAD', # Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.
    19: 'TYPE_LANE_MARKER', # Marking on the road that’s specifically for defining lanes such as single/double white/yellow lines.
    20: 'TYPE_OTHER_GROUND', # Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    21: 'TYPE_WALKABLE', # Most horizontal surface that’s not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    22: 'TYPE_SIDEWALK', # Nicely paved walkable surface when pedestrians most likely to walk on.
}

    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    mean_ious = sum(mean_ious[1:]) / len(mean_ious[1:])
    dic['mean'] = np.round(mean_ious, 3)
    x.add_row(['mean', np.round(mean_ious, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x
    
def format_SSC_results_dg(mean_ious, return_dic=False):
    class_map = {
        0:'Other',
        1:'Driveable Ground',
        2:'Structure',
        3:'Vehicle',
        4:'Nature',
        5:'Living Being',
        6:'Movable objects',
        7:'Other Ground',
            }
    
    x = PrettyTable()
    x.field_names = ['class', 'IoU']
    class_names = list(class_map.values())
    class_ious = mean_ious
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_ious):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])
    
    mean_ious = sum(mean_ious[1:]) / len(mean_ious[1:])
    dic['mean'] = np.round(mean_ious, 3)
    x.add_row(['mean', np.round(mean_ious, 3)])
    
    if return_dic:
        return x, dic 
    else:
        return x


def format_vel_results(mean_epe, return_dic=False):
    class_map = {
        0: 'barrier',
        1: 'bicycle',
        2: 'bus',
        3: 'car',
        4: 'construction_vehicle',
        5: 'motorcycle',
        6: 'pedestrian',
        7: 'traffic_cone',
        8: 'trailer',
        9: 'truck',
    }
    x = PrettyTable()
    x.field_names = ['class', 'EPE']
    class_names = list(class_map.values())
    class_epes = mean_epe
    dic = {}
    
    for cls_name, cls_iou in zip(class_names, class_epes):
        dic[cls_name] = np.round(cls_iou, 3)
        x.add_row([cls_name, np.round(cls_iou, 3)])

    mean_all_epe = mean_epe.mean()
    dic['mean'] = np.round(mean_all_epe, 3)
    x.add_row(['mean', np.round(mean_all_epe, 3)])
    if return_dic:
        return x, dic 
    else:
        return x