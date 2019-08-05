import random

import torch
import torch.nn as nn

from models.criteria.sigmoid_criterion import SigmoidCriterion


class SoftmaxCriterion(SigmoidCriterion):
    def __init__(self):
        super(SoftmaxCriterion, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, a, target):
        a, target = self.process_tensors(a, target)

        b = target.shape[0]
        oldsoftmax_target = torch.LongTensor(b).zero_()
        for i in range(b):
            if target[i].sum() == 0:
                oldsoftmax_target[i] = target.shape[1]
            else:
                oldsoftmax_target[i] = random.choice(target[i].nonzero())
        # target = target.max(1)[0]
        softmax_target = target.nonzero()[:, 1].long()

        loss = self.loss(a, softmax_target.cuda())

        return loss
