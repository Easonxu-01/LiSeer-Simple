'''
Author: EASON XU
Date: 2025-01-19 21:44:44
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-03-03 16:59:57
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/core/evaluation/eval_hooks.py
'''

# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import os.path as osp
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import EvalHook as BaseEvalHook


class OccEvalHook(BaseEvalHook):
    def __init__(self, *args,  **kwargs):
        super(OccEvalHook, self).__init__(*args, **kwargs)  
            
    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from projects.unilidar_plugin.occupancy.apis.test import custom_single_gpu_test
        results = custom_single_gpu_test(runner.model, self.dataloader, show=False)
        
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)
            
            
class OccDistEvalHook(BaseDistEvalHook):
    def __init__(self, *args,  **kwargs):
        super(OccDistEvalHook, self).__init__(*args, **kwargs)       

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    rank = dist.get_rank()
                    # print(f"Rank {rank}: running_var shape = {module.running_var.shape}, dtype = {module.running_var.dtype}, device = {module.running_var.device}")
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from projects.unilidar_plugin.occupancy.apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
  
