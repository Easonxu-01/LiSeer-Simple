from mmcv import runner
from mmcv.runner import HOOKS
from mmcv.runner.hooks.hook import Hook
import torch


@HOOKS.register_module()
class LiDARCurriculumHook(Hook):

    def __init__(
        self,
        schedule,
        alpha_key='sensor_mae_total',   # sensor-MAE log key emitted by the head
        alpha_step=0.04,                 # step size per adjustment
        alpha_min=0.0,
        alpha_max=1.0,
        alpha_target_low=0.5,           # MAE below this raises alpha
        alpha_target_high=0.6,          # MAE above this lowers alpha
        alpha_momentum=0.9,              # EMA momentum
        # Consistency-loss warmup schedule.
        consistency_start_iter=1000,
        consistency_warmup_iters=5000,
        **kwargs
    ):
        """
        schedule: [(start_epoch, end_epoch, cfg_dict), ...] where cfg_dict is::

        {
            'num_of_beams': (low, high),
            'horizontal_angular_resolution': (low, high),
            'lower_vertical_field_of_view_bound': (low, high),
            'upper_vertical_field_of_view_bound': (low, high),
            'LiDAR_height': (low, high)
        }
        """
        self.schedule = schedule

        # Dynamic FiLM alpha.
        self.alpha_key = 'sensor_mae_total'
        self.alpha_step = alpha_step
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_target_low = alpha_target_low
        self.alpha_target_high = alpha_target_high
        self.alpha_momentum = alpha_momentum

        self.ema_mae = None

        # consistency schedule
        self.consistency_start_iter = consistency_start_iter
        self.consistency_warmup_iters = consistency_warmup_iters

    def compute_mean_std(self, cfg_range):
        """Mean/std of the LiDAR parameter ranges for one curriculum stage."""
        params = [
            'num_of_beams',
            'LiDAR_height',
            'horizontal_angular_resolution',
            'lower_vertical_field_of_view_bound',
            'upper_vertical_field_of_view_bound',
        ]

        mean_vals = []
        std_vals = []

        for p in params:
            a, b = cfg_range[p]
            mean = 0.5 * (a + b)
            std = (b - a) / (12 ** 0.5)  # standard deviation of U(a, b)
            mean_vals.append(mean)
            std_vals.append(std)

        mean = torch.tensor(mean_vals, dtype=torch.float32)
        std = torch.tensor(std_vals, dtype=torch.float32)
        return mean, std

    def before_train_epoch(self, runner):
        """Advance the LiDAR sampling ranges and the sensor mean/std for this epoch."""
        epoch = runner.epoch
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        for start, end, cfg in self.schedule:
            if start <= epoch < end:
                # 1. Sampler ranges.
                model.sensor_config.num_of_beams = cfg['num_of_beams']
                model.sensor_config.horizontal_angular_resolution = cfg['horizontal_angular_resolution']
                model.sensor_config.lower_vertical_field_of_view_bound = cfg['lower_vertical_field_of_view_bound']
                model.sensor_config.upper_vertical_field_of_view_bound = cfg['upper_vertical_field_of_view_bound']
                model.sensor_config.LiDAR_height = cfg['LiDAR_height']
                # 2. Normalisation statistics used by the sensor head.
                mean, std = self.compute_mean_std(cfg)
                head = getattr(model, 'tpv_aggregator', None)
                if head is not None and hasattr(head, 'update_sensor_stats'):
                    head.update_sensor_stats(mean, std)
                    runner.logger.info(
                        f"[Epoch {epoch}] Updated Sensor Mean/Std:\n"
                        f"  mean={mean.tolist()}\n"
                        f"  std ={std.tolist()}"
                    )
                runner.logger.info(
                    f"[Epoch {epoch}] Updated Sensor Sampling Config:\n{model.sensor_config}"
                )
                break


    def after_train_iter(self, runner):
        """Adapt the FiLM mixing coefficient and the consistency-loss weight."""

        cur_mae = runner.log_buffer.val_history.get(self.alpha_key, {})
        if cur_mae is None:
            return  # no sensor MAE logged this iteration
        else:
            cur_mae = float(cur_mae[-1])
            if self.ema_mae is None:
                self.ema_mae = cur_mae
            else:
                m = self.alpha_momentum
                self.ema_mae = m * self.ema_mae + (1 - m) * cur_mae
            model = runner.model
            if hasattr(model, 'module'):
                model = model.module
            head = getattr(model, 'tpv_aggregator', None)
            if head is None or not hasattr(head, 'set_film_alpha'):
                return
            alpha = float(getattr(head, 'film_alpha', 0.0))
            if self.ema_mae < self.alpha_target_low:
                alpha = min(alpha + self.alpha_step, self.alpha_max)
            elif self.ema_mae > self.alpha_target_high:
                alpha = max(alpha - self.alpha_step, self.alpha_min)
            head.set_film_alpha(alpha)
            runner.log_buffer.update(
                {'film_alpha': alpha, 'ema_sensor_mae': self.ema_mae},
                runner.outputs.get('num_samples', 1)
            )

        # Consistency-loss warmup.
        cur_iter = runner.iter

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        head = getattr(model, 'tpv_aggregator', None)
        if head is None or not hasattr(head, 'consistency_loss_weight'):
            return

        base_w = 1.0

        if cur_iter < self.consistency_start_iter:
            cur_w = 1e-3
        elif cur_iter < self.consistency_start_iter + self.consistency_warmup_iters:
            progress = (
                (cur_iter - self.consistency_start_iter)
                / float(self.consistency_warmup_iters)
            )
            cur_w = base_w * progress
        else:
            cur_w = base_w

        head.consistency_loss_weight = cur_w

        runner.log_buffer.update(
            {'consistency_weight': cur_w},
            runner.outputs.get('num_samples', 1)
        )
