import horovod.torch as hvd
import torch

import datasets


def get_dataloader(dataset, batch_size=1, num_workers=4, distributed=False):

    if distributed:
        sampler = get_distributed_sampler(dataset, distributed)
        args = {'sampler': sampler}
    else:
        args = {'shuffle': True if dataset.mode == 'train' else False}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, **args,
        num_workers=num_workers, pin_memory=True)

    return loader


def get_distributed_sampler(dataset, distributed=False):
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank())

    return sampler


def get_dataset(dataset_name, **kwargs):
    """
    Args:
        dataset_name : String or tuple of strings to be concatenated.
                       E.g.: ('charades', ), 'kinetics', ('charades', 'stream').
    """
    dataset_name = (dataset_name, ) if isinstance(dataset_name, str) else dataset_name
    Dataset = getattr(datasets, ''.join(name.capitalize() for name in dataset_name))
    dataset = Dataset(**kwargs)

    return dataset
