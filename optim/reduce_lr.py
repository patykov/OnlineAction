import horovod.torch as hvd
import torch
from torch import optim

from optim.lr_scheduler import LRScheduler


class ReduceLR(LRScheduler):
    def __init__(self, lr_config):
        super().__init__(lr_config)
        self.initial_lr = lr_config['initial_lr']

    def _get_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    def broadcast_scheduler_state(self, root_rank):
        state = self.scheduler.state_dict()
        state['last_epoch'] = hvd.broadcast(
            torch.tensor(state['last_epoch']), root_rank=root_rank, name='last_epoch').item()
        state['num_bad_epochs'] = hvd.broadcast(
            torch.tensor(state['num_bad_epochs']),
            root_rank=root_rank, name='num_bad_epochs').item()
        self.scheduler.load_state_dict(state)
