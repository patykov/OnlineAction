import torch.nn as nn


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, a, target, meta, synchronous=False):
        # return a, loss, target
        raise NotImplementedError()
