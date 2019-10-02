
from importlib import import_module

import horovod.torch as hvd
from torch import optim


def get_optimizer(model, lr_scheduler, weight_decay, distributed=False):
    lr_type = lr_scheduler['type']
    lr_scheduler = getattr(import_module(
        'optim.' + lr_type), lr_type.capitalize().replace('_lr', 'LR'))(lr_scheduler['params'])

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_scheduler.initial_lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=weight_decay)
    if distributed:
        optimizer = hvd.DistributedOptimizer(optimizer,
                                             named_parameters=filter(lambda p: p[1].requires_grad,
                                                                     model.named_parameters()))
    lr_scheduler.set_scheduler(optimizer)

    return optimizer, lr_scheduler
