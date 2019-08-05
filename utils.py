import json
import logging
from bisect import bisect_right

import horovod.torch as hvd
import numpy as np
import torch
from torch import optim

import datasets


class LRLambda:
    def __init__(self, gamma, milestones):
        self.gamma = gamma
        self.milestones = milestones

    def __call__(self, epoch):
        return self.gamma[bisect_right(self.milestones, epoch)]


def get_optimizer(model, lr_config, weight_decay, distributed=False):
    milestones = np.cumsum([i[0] for i in lr_config]).astype('int').tolist()
    num_epochs = milestones[-1]
    milestones = milestones[:-1]
    lrs = [i[1] for i in lr_config]
    initial_lr = lrs[0]
    gamma = [l / initial_lr for l in lrs]  # Multiplicative factors

    lr_lambda = LRLambda(gamma, milestones)

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
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return num_epochs, optimizer, scheduler


def broadcast_scheduler_state(scheduler, root_rank):
    import horovod.torch as hvd
    state = scheduler.state_dict()
    state['last_epoch'] = hvd.broadcast(
        torch.tensor(state['last_epoch']), root_rank=root_rank, name='last_epoch').item()
    state['base_lrs'] = hvd.broadcast(
        torch.tensor(state['base_lrs']), root_rank=root_rank, name='base_lrs').tolist()
    state['lr_lambdas'] = [{
        k: hvd.broadcast(torch.tensor(v), root_rank=root_rank, name=k + str(i)).tolist()
        for k, v in lr_lambda.items()
    } for i, lr_lambda in enumerate(state['lr_lambdas'])]
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
    with open(json_file, 'r') as f:
        data = json.load(f)
    default_values = {
        'freeze': False,
        'sample_frames': 0,
        'nonlocal': True,
        'weight_decay': 1e-4,
        'learning_rate': [[10, 0.1], [10, 0.01]],
        'batch_size': 8
    }
    default_values = recursive_update(default_values, data)

    return default_values


def get_dataloaders(dataset, train_file, val_file, train_data, val_data, batch_size,
                    sample_frames=8, num_workers=4, distributed=False, causal=False):
    Dataset = getattr(datasets, dataset.capitalize())

    train_dataset = Dataset(train_data, train_file, sample_frames=sample_frames,
                            mode='train', causal=causal)
    val_dataset = Dataset(val_data, val_file, sample_frames=sample_frames,
                          mode='val', causal=causal, test_clips=1)

    if distributed:
        import horovod.torch as hvd
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        train_args = {'sampler': train_sampler}
        val_args = {'sampler': val_sampler}
    else:
        train_args = {'shuffle': True}
        val_args = {'shuffle': False}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, **train_args,
        num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, **val_args,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


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
