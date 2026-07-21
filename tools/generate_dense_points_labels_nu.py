import os
import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from scipy.spatial import cKDTree
from collections import Counter

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
    return point_labels.reshape(-1, 1).astype(np.uint8)

def process_nuscenes_dataset(nuscenes_dir, version='v1.0-trainval'):
    output_base_dir = "/data/Datasets/nuscenes"
    os.makedirs(output_base_dir, exist_ok=True)
    output_dir = os.path.join(output_base_dir, 'lidarseg_refine', version)
    os.makedirs(output_dir, exist_ok=True)

    nusc = NuScenes(version=version, dataroot=nuscenes_dir, verbose=True)
    for sample in nusc.sample:
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_file = os.path.join(nuscenes_dir, lidar_data['filename'])
        label_file = os.path.join(nuscenes_dir, 'lidarseg', version, f"{lidar_token}_lidarseg.bin")
        refine_file = lidar_file.replace('.bin', '.ply')
        refine_file = refine_file.replace('LIDAR_TOP', 'LIDAR_TOP/refine', 1)
        output_file = os.path.join(output_dir, f"{lidar_token}_lidarseg.bin")

        if os.path.exists(output_file):
            print(f"Skip existing: {lidar_token}")
            continue

        if not os.path.exists(refine_file) or not os.path.exists(label_file):
            print(f"Missing file, skip: {lidar_token}")
            continue

        pc = np.frombuffer(open(lidar_file, "rb").read(), dtype=np.float32)
        original_points = pc.reshape(-1, 5)[:, :3]
        original_labels = np.fromfile(label_file, dtype=np.uint8)
        pcd = o3d.io.read_point_cloud(refine_file)
        dense_points = np.asarray(pcd.points)

        # Propagate labels from the sparse scan onto the densified cloud.
        try:
            dense_labels = generate_dense_labels(original_points, original_labels, dense_points)
            dense_labels.tofile(output_file)
            print(f"Processed: {lidar_token}")
        except Exception as e:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            error_msg = f"[{timestamp}] Error processing {lidar_token}, Error: {str(e)}"
            print(error_msg)

            # Record the failure and move on to the next file.
            with open("failed_dense_labels.txt", "a") as f:
                f.write(f"{timestamp},{lidar_token},{str(e)}\n")
            continue

if __name__ == "__main__":
    nuscenes_dir = "/home/eason/workspace_perception/UniLiDAR/data/nuscenes"
    process_nuscenes_dataset(nuscenes_dir)
    print("Dense labels generation completed.")
