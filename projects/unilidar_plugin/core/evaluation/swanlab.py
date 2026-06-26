'''
Author: EASON XU
Date: 2025-07-16 06:49:55
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2025-07-16 06:49:58
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/core/evaluation/swanlab.py
'''
# ---------------------------------------------
#  Created by Shuocheng Yang, 2025/06/21
# ---------------------------------------------

from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module()
class SwanlabLoggerHook(LoggerHook):

    def __init__(self,
                 init_kwargs=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 commit=True,
                 by_epoch=True,
                 with_step=True):
        super(SwanlabLoggerHook, self).__init__(interval, ignore_last,
                                              reset_flag, by_epoch)
        self.import_swanlab()
        self.init_kwargs = init_kwargs
        self.commit = commit
        self.with_step = with_step

    def import_swanlab(self):
        try:
            import swanlab
        except ImportError:
            raise ImportError('Please run "pip install swanlab" to install SwanLab.')
        self.swanlab = swanlab

    @master_only
    def before_run(self, runner):
        super(SwanlabLoggerHook, self).before_run(runner)
        if self.swanlab is None:
            self.import_swanlab()
        if self.init_kwargs:
            self.swanlab.init(**self.init_kwargs)
        else:
            self.swanlab.init()

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            step = self.get_iter(runner) if self.with_step else None
            for k, v in tags.items():
                self.swanlab.log({k: v}, step=step)

    @master_only
    def after_run(self, runner):
        self.swanlab.finish()