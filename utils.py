import json
import logging
import os
import sys
from bisect import bisect_right

import horovod.torch as hvd
import numpy as np
import torch
from torch import optim


class LRLambda:
    def __init__(self, gamma, milestones):
        self.gamma = gamma
        self.milestones = milestones

    def __call__(self, epoch):
        return self.gamma[bisect_right(self.milestones, epoch)]


def get_optimizer(model, lr_scheduler, weight_decay, distributed=False):
    initial_lr, lr_scheduler = getattr(
        sys.modules[__name__], 'get_' + lr_scheduler['type'] + '_scheduler')(lr_scheduler['params'])

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay)
    if distributed:
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=filter(lambda p: p[1].requires_grad,
                                                                     model.named_parameters()))
    scheduler = lr_scheduler(optimizer)

    return optimizer, scheduler


def get_lambdaLR_scheduler(lr_config):
    milestones = np.cumsum([i[0] for i in lr_config['learning_rate']]).astype('int').tolist()
    milestones = milestones[:-1]
    lrs = [i[1] for i in lr_config['learning_rate']]
    initial_lr = lrs[0]
    gamma = [l / initial_lr for l in lrs]  # Multiplicative factors

    lr_lambda = LRLambda(gamma, milestones)

    def scheduler(optimizer):
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return initial_lr, scheduler


def get_reduceLR_scheduler(lr_config):

    def scheduler(optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    return lr_config['initial_lr'], scheduler


def get_cosineLR_scheduler(lr_config):

    def scheduler(optimizer):
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_config['T_max'])

    return lr_config['initial_lr'], scheduler


def broadcast_scheduler_state(scheduler, root_rank):
    state = scheduler.state_dict()
    state['last_epoch'] = hvd.broadcast(
        torch.tensor(state['last_epoch']), root_rank=root_rank, name='last_epoch').item()

    if isinstance(scheduler, optim.lr_scheduler.LambdaLR):
        state['base_lrs'] = hvd.broadcast(
            torch.tensor(state['base_lrs']), root_rank=root_rank, name='base_lrs').tolist()
        state['lr_lambdas'] = [{
            k: hvd.broadcast(torch.tensor(v), root_rank=root_rank, name=k + str(i)).tolist()
            for k, v in lr_lambda.items()
        } for i, lr_lambda in enumerate(state['lr_lambdas'])]

    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        state['num_bad_epochs'] = hvd.broadcast(
            torch.tensor(state['num_bad_epochs']),
            root_rank=root_rank, name='num_bad_epochs').item()

    scheduler.load_state_dict(state)


def recursive_update(d, u):
    '''
    Recursively update values in a dict (and dicts inside)
    '''
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        elif v is not None:
            d[k] = v
    return d


def parse_json(json_file):
    if not os.path.isabs(json_file):
        json_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'config_files', json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)
    default_values = {
        'nonlocal': True,
        'weight_decay': 1e-4,
        'learning_scheduler': {
            'type': 'reduceLR',
            'params': {
                'initial_lr': 0.001
            }
        },
        'num_epochs': 30,
        'batch_size': 8
    }
    default_values = recursive_update(default_values, data)

    return default_values


def setup_logger(logger_name, log_file):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    streamHandler.setLevel(logging.INFO)

    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
