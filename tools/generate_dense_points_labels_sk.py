'''
Author: EASON XU
Date: 2025-05-07 09:40:46
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-05-08 03:51:42
Description: 头部注释
FilePath: /UniLiDAR/generate_dense_points_labels.py
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
    seq, bin_file, root_dir, output_base_dir = args
    logger = logging.getLogger()

    seq_path = os.path.join(root_dir, "sequences", seq)
    velodyne_path = os.path.join(seq_path, "velodyne")
    label_path = os.path.join(seq_path, "labels")
    refine_path = os.path.join(seq_path, "refine")
    output_path = os.path.join(output_base_dir, "sequences", seq, "dense_labels")
    os.makedirs(output_path, exist_ok=True)

    bin_file_path = os.path.join(velodyne_path, bin_file)
    label_file_path = os.path.join(label_path, bin_file.replace(".bin", ".label"))
    refine_file_path = os.path.join(refine_path, bin_file.replace(".bin", ".ply"))
    output_file_path = os.path.join(output_path, bin_file.replace(".bin", ".label"))

    if not os.path.exists(refine_file_path):
        logger.warning(f"Refine file not found: {refine_file_path}")
        return

    try:
        original_points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        original_labels = np.fromfile(label_file_path, dtype=np.uint32)
        pcd = o3d.io.read_point_cloud(refine_file_path)
        dense_points = np.asarray(pcd.points)

        dense_labels = generate_dense_labels(original_points, original_labels, dense_points)
        dense_labels.tofile(output_file_path)
        print(f"Processed: Sequence {seq}, File: {bin_file}")
    except Exception as e:
        logger.error(f"Error processing file {bin_file}: {e}")

def process_semantickitti_dataset(root_dir):
    sequences = [f"{i:02d}" for i in range(11)]
    output_base_dir = "/data/Datasets/semantickitti"
    os.makedirs(output_base_dir, exist_ok=True)

    tasks = []
    for seq in sequences:
        velodyne_path = os.path.join(root_dir, "sequences", seq, "velodyne")
        bin_files = sorted(os.listdir(velodyne_path))
        for bin_file in bin_files:
            tasks.append((seq, bin_file, root_dir, output_base_dir))

    queue = Queue()
    listener = get_context('spawn').Process(target=listener_process, args=(queue,))
    listener.start()

    with get_context('spawn').Pool(processes=cpu_count(), initializer=init_worker, initargs=(queue,)) as pool:
        pool.map(process_file, tasks)

    queue.put(None)
    listener.join()

if __name__ == "__main__":
    root_dir = "/home/eason/workspace_perception/UniLiDAR/data/semantickitti"
    process_semantickitti_dataset(root_dir)
    print("Dense labels generation completed.")
