'''
Author: EASON XU
Date: 2025-07-10 06:17:10
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-07-22 05:59:52
Description: 头部注释
FilePath: /UniLiDAR/generate_dense_points_labels_waymo.py
'''
import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from collections import Counter
from multiprocessing import Pool, cpu_count, get_context, Queue
import logging
import logging.handlers
from natsort import natsorted
import gc

log_queue = None

def listener_process(queue):
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)

def init_worker(queue):
    global log_queue
    log_queue = queue
    handler = logging.handlers.QueueHandler(log_queue)
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(logging.INFO)

def generate_dense_labels(original_points, original_labels, dense_points, voxel_size=0.2):
    def voxelize(points):
        return np.floor(points / voxel_size).astype(np.int32)

    def majority_vote(labels):
        if len(labels) == 0:
            return -1
        return Counter(labels).most_common(1)[0][0]

    voxel_indices = voxelize(original_points)
    voxel_label_dict = {}
    voxel_dict = {}
    for voxel, label in zip(map(tuple, voxel_indices), original_labels):
        voxel_dict.setdefault(voxel, []).append(label)
    for voxel, labels in voxel_dict.items():
        voxel_label_dict[voxel] = majority_vote(labels)

    dense_voxel_indices = voxelize(dense_points)
    unique_dense_voxels = np.unique(dense_voxel_indices, axis=0)

    filled_voxels = np.array(list(voxel_label_dict.keys()))
    filled_labels = np.array(list(voxel_label_dict.values()))
    tree = cKDTree(filled_voxels)

    dense_voxel_label_dict = {}
    for voxel in map(tuple, unique_dense_voxels):
        if voxel in voxel_label_dict:
            dense_voxel_label_dict[voxel] = voxel_label_dict[voxel]
        else:
            dist, idx = tree.query(voxel, k=1)
            dense_voxel_label_dict[voxel] = filled_labels[idx]

    point_labels = np.array([dense_voxel_label_dict[tuple(voxel)] for voxel in dense_voxel_indices])
    return point_labels.reshape(-1, 1).astype(np.uint32)

def process_file(args):
    bin_file, root_dir, output_base_dir = args
    logger = logging.getLogger()

    velodyne_path = os.path.join(root_dir, "velodyne")
    label_path = os.path.join(root_dir, "velodyne")
    refine_path = os.path.join(root_dir, "refine")
    # output_path = os.path.join(output_base_dir, "dense_labels")
    output_path = output_base_dir
    os.makedirs(output_path, exist_ok=True)

    bin_file_path = os.path.join(velodyne_path, bin_file)
    label_file_path = os.path.join(label_path, bin_file.replace(".bin", ".label"))
    refine_file_path = os.path.join(refine_path, bin_file.replace(".bin", ".ply"))
    output_file_path = os.path.join(output_path, bin_file.replace(".bin", ".label"))

    if os.path.exists(output_file_path):
        logger.info(f"Output file already exists, skipping: {output_file_path}")
        return

    if not os.path.exists(refine_file_path):
        logger.warning(f"Refine file not found: {refine_file_path}")
        return

    try:
        original_points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 6)[:, :3]
        original_labels = np.fromfile(label_file_path, dtype=np.int32).reshape(-1, 2)[:, 1]
        try:
            assert original_labels.shape[0] == original_points.shape[0]
        except AssertionError:
            logger.error(f"Shape mismatch in file {bin_file}: original_labels.shape[0]={original_labels.shape[0]}, original_points.shape[0]={original_points.shape[0]}")
            return
        pcd = o3d.io.read_point_cloud(refine_file_path)
        dense_points = np.asarray(pcd.points)
        dense_labels = generate_dense_labels(original_points, original_labels, dense_points)
        dense_labels.tofile(output_file_path)
        print(f"Processed: File: {bin_file}")
    except Exception as e:
        logger.error(f"Error processing file {bin_file}: {e}")
    finally:
        # 主动清理大对象和垃圾回收
        for var in ["original_points", "original_labels", "pcd", "dense_points", "dense_labels"]:
            if var in locals():
                del locals()[var]
        gc.collect()

def process_waymo_dataset(root_dir, log_queue):
    output_base_dir = "/data/Datasets/Waymo/raw/Waymo/Perception_Dataset/waymo_open_dataset_v_1_4_0/kitti_format/training/dense_labels"
    os.makedirs(output_base_dir, exist_ok=True)
    tasks = []
    velodyne_path = os.path.join(root_dir, "velodyne")
    # Load annotated indices from seg_annotated_indices.txt
    annotated_indices_path = 'seg_annotated_indices.txt'
    with open(annotated_indices_path, 'r') as f:
        annotated_files = set(line.strip() for line in f.readlines())
    
    print(f"Loaded {len(annotated_files)} annotated files from seg_annotated_indices.txt")

    # Get all bin files and filter by annotated indices
    all_bin_files = natsorted([f for f in os.listdir(velodyne_path) if f.endswith('.bin')])
    annotated_bin_files = [f for f in all_bin_files if f in annotated_files]
    
    print(f" Found {len(annotated_bin_files)} annotated files out of {len(all_bin_files)} total files")
    for bin_file in annotated_bin_files:
        # 检查输出文件是否已存在，若存在则跳过
        output_file_path = os.path.join(output_base_dir, bin_file.replace(".bin", ".label"))
        if os.path.exists(output_file_path):
            continue
        tasks.append((bin_file, root_dir, output_base_dir))

    print(f"Start multi-process processing of {len(tasks)} files...")
    from multiprocessing import get_context, cpu_count
    with get_context("spawn").Pool(processes=cpu_count(), initializer=init_worker, initargs=(log_queue,)) as pool:
        for idx, _ in enumerate(pool.imap_unordered(process_file, tasks), 1):
            print(f"[{idx}/{len(tasks)}] Processed.")
    print("All files processed.")

if __name__ == "__main__":
    from multiprocessing import Process, Queue
    log_queue = Queue()
    listener = Process(target=listener_process, args=(log_queue,))
    listener.start()
    root_dir = "/home/eason/workspace_perception/UniLiDAR/data/waymo/kitti_format/training"
    process_waymo_dataset(root_dir, log_queue)
    log_queue.put(None)
    listener.join()
    print("Dense labels generation completed.")
