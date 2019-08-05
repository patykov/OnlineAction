import torch.nn as nn

from models.criteria.criterion import Criterion
from models.layers.balance_labels import BalanceLabels


class SigmoidCriterion(Criterion):
    def __init__(self, balance_loss):
        super(SigmoidCriterion, self).__init__()
        self.balance_loss = balance_loss
        self.balance_labels = BalanceLabels()
        self.loss = nn.BCEWithLogitsLoss()

    def process_tensors(self, a, target, balance=False):
        target = target.float()
        if balance and self.training:
            a = self.balance_labels(a, target)
        return a, target

    def forward(self, a, target):
        a, target = self.process_tensors(a, target, self.balance_loss)
        loss = self.loss(a, target)

        return loss
