'''
Author: EASON XU
Date: 2026-01-12 17:44:31
LastEditors: EASON XU
Version: Do not edit
LastEditTime: 2026-01-12 17:52:45
Description: 头部注释
FilePath: /UniLiDAR/projects/unilidar_plugin/pointocc_model/distill_epoch_runner.py
'''
import torch
from mmcv.runner import EpochBasedRunner, RUNNERS


@RUNNERS.register_module()
class DistillEpochBasedRunner(EpochBasedRunner):
    """Epoch-based runner that supports a frozen teacher model for distillation.

    使用方式（示例）::

        runner = DistillEpochBasedRunner(
            model=student,
            model_t=teacher,
            optimizer=optimizer,
            work_dir=work_dir,
            logger=logger,
            meta=meta)

    约定:
        - ``model`` 为 student，只对其构建 optimizer 并反向传播。
        - ``model_t`` 为 teacher，通常在外部已经 ``eval()`` 且 ``requires_grad=False``。
        - student 的 ``train_step`` 需要支持形如:

              def train_step(self, data_batch, optimizer, model_t=None, **kwargs):
                  ...

          本 runner 会在训练时将 ``model_t`` 以关键字参数传入。
    """

    def __init__(self, model, optimizer=None, model_t=None, **kwargs):
        super().__init__(model=model, optimizer=optimizer, **kwargs)
        self.model_t = model_t

    def run_iter(self, data_batch, train_mode, **kwargs):
        """Run a single iteration.

        在训练模式下，调用:
            model.train_step(data_batch, optimizer, model_t=self.model_t, **kwargs)
        在验证/测试模式下，调用:
            model.val_step(data_batch, **kwargs)
        """
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            # student 负责计算损失并反向；teacher 通常在模型内部使用。
            if self.model_t is not None:
                outputs = self.model.train_step(
                    data_batch, self.optimizer, model_t=self.model_t, **kwargs)
            else:
                outputs = self.model.train_step(
                    data_batch, self.optimizer, **kwargs)
        else:
            outputs = self.model.val_step(data_batch, **kwargs)

        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs


