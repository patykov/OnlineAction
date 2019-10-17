import horovod.torch as hvd
import torch

import datasets


def get_dataloader(dataset_name, batch_size=1, num_workers=4, distributed=False, **kargs):
    dataset = get_dataset(dataset_name, **kargs)

    if distributed:
        sampler = get_distributed_sampler(dataset_name, distributed, **kargs)
        args = {'sampler': sampler}
    else:
        args = {'shuffle': True if kargs['mode'] == 'train' else False}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, **args,
        num_workers=num_workers, pin_memory=True)

    return loader


def get_distributed_sampler(dataset_name, distributed=False, **kargs):
    dataset = get_dataset(dataset_name, **kargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank())

    return sampler


def get_dataset(dataset_name, **kargs):
    Dataset = getattr(
        datasets, dataset_name.capitalize() if dataset_name[0].islower() else dataset_name)
    dataset = Dataset(**kargs)

    return dataset
