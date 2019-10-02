from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


class LRScheduler(_LRScheduler):
    def __init__(self, lr_config):
        if 'warm_up' in lr_config:
            self.warm_up = lr_config['warm_up']
        else:
            self.warm_up = None

        # 'step_per_iter' tells if the scheduler will take a step each iteration or each epoch
        self.step_per_iter = lr_config['step_per_iter']

    def set_scheduler(self, optimizer):
        self.scheduler = self._get_scheduler(optimizer)
        if self.warm_up:
            self.set_warm_up(optimizer)

    def _get_scheduler(self, optimizer):
        raise NotImplementedError

    def step(self, metrics=None):
        if type(self.scheduler) != ReduceLROnPlateau:
            self.scheduler.step()
        else:
            self.scheduler.step(metrics)

    def set_warm_up(self, optimizer):
        from warmup_scheduler import GradualWarmupScheduler
        after_scheduler = self.scheduler
        scheduler_warmup = GradualWarmupScheduler(
            optimizer, multiplier=self.warm_up['multiplier'],
            total_epoch=self.warm_up['total_epoch'],
            after_scheduler=after_scheduler)

        self.scheduler = scheduler_warmup

    def broadcast_scheduler_state(self, root_rank):
        raise NotImplementedError

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

    @property
    def last_epoch(self):
        return self.scheduler.last_epoch
