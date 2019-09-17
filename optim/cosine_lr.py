import horovod.torch as hvd
import torch
from torch import optim

from optim.lr_scheduler import LRScheduler


class CosineLR(LRScheduler):
    def __init__(self, lr_config):
        super().__init__(lr_config)
        self.initial_lr = lr_config['initial_lr']
        self.T_max = lr_config['T_max']

    def _get_scheduler(self, optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max)

    def broadcast_scheduler_state(self, root_rank):
        state = self.scheduler.state_dict()
        state['last_epoch'] = hvd.broadcast(
            torch.tensor(state['last_epoch']), root_rank=root_rank, name='last_epoch').item()
        self.scheduler.load_state_dict(state)
