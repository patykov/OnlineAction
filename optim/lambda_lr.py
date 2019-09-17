from bisect import bisect_right

import horovod.torch as hvd
import numpy as np
import torch
from torch import optim

from optim.lr_scheduler import LRScheduler


class GammaScheduler:
    def __init__(self, gamma, milestones):
        self.gamma = gamma
        self.milestones = milestones

    def __call__(self, epoch):
        return self.gamma[bisect_right(self.milestones, epoch)]


class LambdaLR(LRScheduler):
    def __init__(self, lr_config):
        super().__init__(lr_config)
        milestones = np.cumsum([i[0] for i in lr_config['learning_rate']]).astype('int').tolist()
        milestones = milestones[:-1]
        lrs = [i[1] for i in lr_config['learning_rate']]

        self.initial_lr = lrs[0]

        gamma = [l / self.initial_lr for l in lrs]  # Multiplicative factors
        self.lr_lambda = GammaScheduler(gamma, milestones)

    def _get_scheduler(self, optimizer):
        return optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

    def broadcast_scheduler_state(self, root_rank):
        state = self.scheduler.state_dict()
        state['last_epoch'] = hvd.broadcast(
            torch.tensor(state['last_epoch']), root_rank=root_rank, name='last_epoch').item()
        state['base_lrs'] = hvd.broadcast(
            torch.tensor(state['base_lrs']), root_rank=root_rank, name='base_lrs').tolist()
        state['lr_lambdas'] = [{
            k: hvd.broadcast(torch.tensor(v), root_rank=root_rank, name=k + str(i)).tolist()
            for k, v in lr_lambda.items()
        } for i, lr_lambda in enumerate(state['lr_lambdas'])]
        self.scheduler.load_state_dict(state)
