import torch
from mmcv.runner import EpochBasedRunner, RUNNERS


@RUNNERS.register_module()
class DistillEpochBasedRunner(EpochBasedRunner):
    """Epoch-based runner that supports a frozen teacher model for distillation.

    Example::

        runner = DistillEpochBasedRunner(
            model=student,
            model_t=teacher,
            optimizer=optimizer,
            work_dir=work_dir,
            logger=logger,
            meta=meta)

    Contract:
        - ``model`` is the student: it owns the optimizer and receives gradients.
        - ``model_t`` is the teacher, normally already ``eval()`` with
          ``requires_grad=False`` before it is handed over.
        - the student's ``train_step`` must accept ``model_t``::

              def train_step(self, data_batch, optimizer, model_t=None, **kwargs):
                  ...

          The runner passes ``model_t`` as a keyword argument during training.
    """

    def __init__(self, model, optimizer=None, model_t=None, **kwargs):
        super().__init__(model=model, optimizer=optimizer, **kwargs)
        self.model_t = model_t

    def run_iter(self, data_batch, train_mode, **kwargs):
        """Run a single iteration.

        Training calls ``model.train_step(data_batch, optimizer,
        model_t=self.model_t, **kwargs)``; validation and test call
        ``model.val_step(data_batch, **kwargs)``.
        """
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            # The student computes and backpropagates the loss; the teacher is
            # consumed inside the model.
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


