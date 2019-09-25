import horovod.torch as hvd
import torch

import datasets


def get_dataloader(dataset, data_file, data, batch_size, mode, sample_frames=8, num_workers=4,
                   distributed=False, subset=False):
    Dataset = getattr(datasets, dataset.capitalize())

    dataset = Dataset(data, data_file, sample_frames=sample_frames, mode=mode, subset=subset)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank())
        args = {'sampler': sampler}
    else:
        args = {'shuffle': True if mode == 'train' else False}

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, **args,
        num_workers=num_workers, pin_memory=True)

    return loader
