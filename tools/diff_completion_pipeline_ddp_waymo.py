'''
Author: EASON XU
Date: 2024-10-21 14:59:33
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-07-14 02:09:53
Description: 头部注释
FilePath: /lidiff/tools/diff_completion_pipeline_ddp_waymo.py
'''
import numpy as np
import MinkowskiEngine as ME
import torch
import lidiff.models.minkunet as minknet
import open3d as o3d
from diffusers import DPMSolverMultistepScheduler
from pytorch_lightning.core.lightning import LightningModule
import yaml
import os
import tqdm
from natsort import natsorted
import click
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import Process
import gc

class DiffCompletion(LightningModule):
    def __init__(self, diff_path, refine_path, denoising_steps, cond_weight):
        super().__init__()
        ckpt_diff = torch.load(diff_path)
        self.save_hyperparameters(ckpt_diff['hyper_parameters'])
        assert denoising_steps <= self.hparams['diff']['t_steps'], \
        f"The number of denoising steps cannot be bigger than T={self.hparams['diff']['t_steps']} (you've set '-T {denoising_steps}')"

        self.partial_enc = minknet.MinkGlobalEnc(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.model = minknet.MinkUNetDiff(in_channels=3, out_channels=self.hparams['model']['out_dim'])
        self.model_refine = minknet.MinkUNet(in_channels=3, out_channels=3*6)
        self.load_state_dict(ckpt_diff['state_dict'], strict=False)

        ckpt_refine = torch.load(refine_path)
        self.load_state_dict(ckpt_refine['state_dict'], strict=False)

        self.partial_enc.eval()
        self.model.eval()
        self.model_refine.eval()
        self.cuda()

        # for fast sampling
        self.hparams['diff']['s_steps'] = denoising_steps
        self.dpm_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=self.hparams['diff']['t_steps'],
                beta_start=self.hparams['diff']['beta_start'],
                beta_end=self.hparams['diff']['beta_end'],
                beta_schedule='linear',
                algorithm_type='sde-dpmsolver++',
                solver_order=2,
        )
        self.dpm_scheduler.set_timesteps(self.hparams['diff']['s_steps'])
        self.scheduler_to_cuda()

        self.hparams['train']['uncond_w'] = cond_weight
        self.hparams['data']['max_range'] = 50.
        self.w_uncond = self.hparams['train']['uncond_w']
        
        exp_dir = diff_path.split('/')[-1].split('.')[0].replace('=','')  + f'_T{denoising_steps}_s{cond_weight}'
        os.makedirs(f'./results/{exp_dir}', exist_ok=True)
        with open(f'./results/{exp_dir}/exp_config.yaml', 'w+') as exp_config:
            yaml.dump(self.hparams, exp_config)

    def scheduler_to_cuda(self):
        self.dpm_scheduler.timesteps = self.dpm_scheduler.timesteps.cuda()
        self.dpm_scheduler.betas = self.dpm_scheduler.betas.cuda()
        self.dpm_scheduler.alphas = self.dpm_scheduler.alphas.cuda()
        self.dpm_scheduler.alphas_cumprod = self.dpm_scheduler.alphas_cumprod.cuda()
        self.dpm_scheduler.alpha_t = self.dpm_scheduler.alpha_t.cuda()
        self.dpm_scheduler.sigma_t = self.dpm_scheduler.sigma_t.cuda()
        self.dpm_scheduler.lambda_t = self.dpm_scheduler.lambda_t.cuda()
        self.dpm_scheduler.sigmas = self.dpm_scheduler.sigmas.cuda()

    def points_to_tensor(self, points):
        x_feats = ME.utils.batched_coordinates(list(points[:]), dtype=torch.float32, device=self.device)

        x_coord = x_feats.clone()
        x_coord = torch.round(x_coord / self.hparams['data']['resolution'])

        x_t = ME.TensorField(
            features=x_feats[:,1:],
            coordinates=x_coord,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=self.device,
        )

        torch.cuda.empty_cache()

        return x_t                                                                                        

    def reset_partial_pcd(self, x_part, x_uncond):
        x_part = self.points_to_tensor(x_part.F.reshape(1,-1,3).detach())
        x_uncond = self.points_to_tensor(torch.zeros_like(x_part.F.reshape(1,-1,3)))

        return x_part, x_uncond

    def preprocess_scan(self, scan):
        dist = np.sqrt(np.sum((scan)**2, -1))
        scan = scan[(dist < self.hparams['data']['max_range']) & (dist > 3.5)][:,:3]

        # use farthest point sampling with fallback for small point clouds
        pcd_scan = o3d.geometry.PointCloud()
        pcd_scan.points = o3d.utility.Vector3dVector(scan)
        
        # Try 1/10 sampling first, if it fails, use 1/30
        try:
            target_points = int(self.hparams['data']['num_points'] / 10)
            pcd_scan = pcd_scan.farthest_point_down_sample(target_points)
            repeat_factor = 10
        except Exception as e:
            print(f"Warning: 1/10 sampling failed, trying 1/30 sampling. Error: {e}")
            target_points = int(self.hparams['data']['num_points'] / 40)
            pcd_scan = pcd_scan.farthest_point_down_sample(target_points)
            repeat_factor = 40
            
        scan = torch.tensor(np.array(pcd_scan.points)).cuda()
        scan = scan.repeat(repeat_factor, 1)
        scan = scan[None,:,:]
        
        # Clear intermediate variables
        del pcd_scan, dist

        return scan

    def postprocess_scan(self, completed_scan, input_scan):
        dist = np.sqrt(np.sum((completed_scan)**2, -1))
        post_scan = completed_scan[dist < self.hparams['data']['max_range']]
        max_z = input_scan[...,2].max().item()
        min_z = (input_scan[...,2].mean() - 2 * input_scan[...,2].std()).item()

        post_scan = post_scan[(post_scan[:,2] < max_z) & (post_scan[:,2] > min_z)]

        return post_scan

    def complete_scan(self, scan):
        scan = self.preprocess_scan(scan)
        x_feats = scan + torch.randn(scan.shape, device=self.device)
        x_full = self.points_to_tensor(x_feats)
        x_cond = self.points_to_tensor(scan)
        x_uncond = self.points_to_tensor(torch.zeros_like(scan))

        completed_scan = self.completion_loop(scan, x_full, x_cond, x_uncond)
        post_scan = self.postprocess_scan(completed_scan, scan)

        refine_in = self.points_to_tensor(post_scan[None,:,:])
        offset = self.refine_forward(refine_in).reshape(-1,6,3)

        refine_complete_scan = post_scan[:,None,:] + offset.cpu().numpy()
        
        # Clear GPU memory
        del x_feats, x_full, x_cond, x_uncond, completed_scan, refine_in, offset
        torch.cuda.empty_cache()

        return refine_complete_scan.reshape(-1,3), post_scan

    def refine_forward(self, x_in):
        with torch.no_grad():
            offset = self.model_refine(x_in)

        return offset

    def forward(self, x_full, x_full_sparse, x_part, t):
        with torch.no_grad():
            part_feat = self.partial_enc(x_part)
            out = self.model(x_full, x_full_sparse, part_feat, t)

        torch.cuda.empty_cache()
        return out.reshape(t.shape[0],-1,3)

    def classfree_forward(self, x_t, x_cond, x_uncond, t):
        x_t_sparse = x_t.sparse()
        x_cond = self.forward(x_t, x_t_sparse, x_cond, t)            
        x_uncond = self.forward(x_t, x_t_sparse, x_uncond, t)

        return x_uncond + self.w_uncond * (x_cond - x_uncond)

    def completion_loop(self, x_init, x_t, x_cond, x_uncond):
        self.scheduler_to_cuda()

        for t in tqdm.tqdm(range(len(self.dpm_scheduler.timesteps))):
            t = self.dpm_scheduler.timesteps[t].cuda()[None]

            noise_t = self.classfree_forward(x_t, x_cond, x_uncond, t)
            input_noise = x_t.F.reshape(t.shape[0],-1,3) - x_init
            x_t = x_init + self.dpm_scheduler.step(noise_t, t, input_noise)['prev_sample']
            x_t = self.points_to_tensor(x_t)

            x_cond, x_uncond = self.reset_partial_pcd(x_cond, x_uncond)
            
            # Clear intermediate tensors
            del noise_t, input_noise
            torch.cuda.empty_cache()

        return x_t.F.cpu().detach().numpy()

def load_pcd(pcd_file):
    if pcd_file.endswith('.bin'):
        # return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,4))[:,:3]
        return np.fromfile(pcd_file, dtype=np.float32).reshape((-1,6))[:,:3]
    elif pcd_file.endswith('.ply'):
        return np.array(o3d.io.read_point_cloud(pcd_file).points)
    else:
        print(f"Point cloud format '.{pcd_file.split('.')[-1]}' not supported. (supported formats: .bin (kitti format), .ply)")

def setup(rank, world_size):
    """Setup the process group for DDP."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'  # Pick any free port number

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"DDP initialized on rank {rank}")

def cleanup():
    """Cleanup the DDP process group."""
    dist.destroy_process_group()

def inference(rank, diff, refine, denoising_steps, cond_weight, world_size):
    """Inference logic for each process (GPU)."""
    setup(rank, world_size)

    # Initialize model and move it to the appropriate GPU
    device = torch.device(f'cuda:{rank}')
    model = DiffCompletion(
            diff, refine, denoising_steps, cond_weight
        ).to(device)
    model = DDP(model, device_ids=[rank])

    # Process training data
    base_path = './Datasets/Waymo/kitti_format/training'
    velodyne_path = os.path.join(base_path, 'velodyne')
    output_refine_path = os.path.join(base_path, 'refine')
    os.makedirs(output_refine_path, exist_ok=True)

    # Load annotated indices from seg_annotated_indices.txt
    annotated_indices_path = './obj_annotated.txt'
    with open(annotated_indices_path, 'r') as f:
        annotated_files = set(line.strip() for line in f.readlines())
    
    print(f"GPU {rank}: Loaded {len(annotated_files)} annotated files from seg_annotated_indices.txt")

    # Get all bin files and filter by annotated indices
    all_bin_files = natsorted([f for f in os.listdir(velodyne_path) if f.endswith('.bin')])
    annotated_bin_files = [f for f in all_bin_files if f in annotated_files]
    
    print(f"GPU {rank}: Found {len(annotated_bin_files)} annotated files out of {len(all_bin_files)} total files")
    
    # Calculate files per GPU
    files_per_gpu = len(annotated_bin_files) // world_size
    start_idx = rank * files_per_gpu
    end_idx = start_idx + files_per_gpu if rank != world_size - 1 else len(annotated_bin_files)
    
    # Assign files to this GPU
    my_bin_files = annotated_bin_files[start_idx:end_idx]
    
    print(f"GPU {rank}: Processing {len(my_bin_files)} annotated files out of {len(annotated_bin_files)} total annotated files")
    print(f"GPU {rank}: Files range from {start_idx} to {end_idx-1}")

    # Process assigned .bin files
    for bin_file in tqdm.tqdm(my_bin_files, desc=f'GPU {rank} - Training'):
        # Double check if file is in annotated list (safety check)
        if bin_file not in annotated_files:
            print(f'GPU {rank}: {bin_file} not in annotated list, skipping...')
            continue
            
        # Check if target file already exists
        refine_file = os.path.join(output_refine_path, bin_file.replace('.bin', '.ply'))
        if os.path.exists(refine_file):
            print(f'GPU {rank}: {bin_file} already processed, skipping...')
            continue
            
        bin_path = os.path.join(velodyne_path, bin_file)
        points = load_pcd(bin_path)  # Load point cloud as numpy array

        # Perform completion on the GPU
        start = time.time()
        refine_scan, _ = model.module.complete_scan(points)  # Use model.module with DDP
        end = time.time()
        print(f'GPU {rank}: {bin_file} took: {end - start:.2f}s')

        # Save refined point cloud
        pcd_refine = o3d.geometry.PointCloud()
        pcd_refine.points = o3d.utility.Vector3dVector(refine_scan)
        pcd_refine.estimate_normals()
        o3d.io.write_point_cloud(refine_file, pcd_refine)
        
        # Clear memory
        del pcd_refine, refine_scan, points
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection

    cleanup()

@click.command()
@click.option('--diff', '-d', type=str, default='checkpoints/diff_net.ckpt', help='path to the scan sequence')
@click.option('--refine', '-r', type=str, default='checkpoints/refine_net.ckpt', help='path to the scan sequence')
@click.option('--denoising_steps', '-T', type=int, default=50, help='number of denoising steps (default: 50)')
@click.option('--cond_weight', '-s', type=float, default=6.0, help='conditioning weight (default: 6.0)')
@click.option('--world_size', type=int, default=int(torch.cuda.device_count()), help='Number of GPUs to use.')
def main(diff, refine, denoising_steps, cond_weight, world_size):
    # Start the DDP process for each GPU
    import torch.multiprocessing as mp
    mp.spawn(inference, args=(diff, refine, denoising_steps, cond_weight, world_size), nprocs=world_size)

if __name__ == '__main__':
    main()