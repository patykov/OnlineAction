import horovod.torch as hvd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .charades_classify import charades_map


class Metric:
    def __init__(self, name):
        self.aggregate = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.aggregate = 0
        self.count = 0

    @torch.no_grad()
    def add(self, output, target):
        self._add(output, target)

    @property
    def value(self):
        return self.aggregate / self.count

    def _add(self, output, target):
        raise NotImplementedError()


class Accuracy(Metric):
    def _add(self, output, target):
        preds = output.argmax(-1)
        self.aggregate += hvd.allreduce(
            torch.sum(preds == target.type_as(preds)), average=False,
            name=self.name + 'agg').item()
        self.count += hvd.allreduce(
            torch.prod(torch.as_tensor(preds.shape[:2])), average=False,
            name=self.name + 'count').item()

    def __repr__(self):
        return 'accuracy'


class Recall(Metric):
    def __init__(self, threshold, name):
        super().__init__(name)
        self.threshold = threshold

    def _add(self, output, target):
        preds = output > self.threshold
        # Recall is TP / (TP + FN)
        self.aggregate += hvd.allreduce(
            torch.sum(preds * target.type_as(preds)), average=False, name=self.name + 'agg').item()
        self.count += hvd.allreduce(
            torch.tensor(target.size(0)), average=False, name=self.name + 'count').item()

    def __repr__(self):
        return 'recall(thr={:g})'.format(self.threshold)


class Video_Accuracy(Metric):
    def __init__(self):
        super().__init__(name='video_accuracy')
        self.text = '{:^5} | {:^20}\n'.format('Label', 'Top5 predition')
        self.top1_pred = []
        self.top5_pred = []
        self.labels = []

    def _add(self, output, target):
        rst = output.mean(0)
        _, top5_pred = torch.topk(rst, 5)

        target = target['target'][0]

        self.top1_pred.append(top5_pred[0])
        self.top5_pred.append(target if target in top5_pred else top5_pred[0])
        self.labels.append(target)

        self.text += '{:^5} | {:^20}\n'.format(target, np.array2string(
            top5_pred.numpy(), separator=', ')[1:-1])

    def __repr__(self):
        acc1 = get_accuracy(self.top1_pred, self.labels)
        acc5 = get_accuracy(self.top5_pred, self.labels)
        return 'Acc: {:.02f} (top1) / {:.02f} (top5)'.format(acc1, acc5)

    def to_text(self):
        partial_results = self.text[:-2]  # Removing last '\n'
        self.text = ''
        return partial_results


class Video_mAP(Metric):
    def __init__(self):
        self.am = AverageMeter()
        self.predictions = []
        self.targets = []
        self.text = ''

    def _add(self, output, target):
        rst = output.mean(0)
        prediction = torch.sigmoid(rst)

        self.targets.append(target['target'].numpy()[0])
        self.predictions.append(prediction)

        self.text += '{} {}\n'.format(target['video_path'][0], np.array2string(
            prediction.numpy(), separator=' ',
            formatter={'float_kind': lambda x: '%.8f' % x})[1:-1].replace('\n', ''))

    def __repr__(self):
        mAP, _, ap = charades_map(np.vstack(self.predictions), np.vstack(self.targets))
        return 'mAP: {:.5f}'.format(mAP)

    def to_text(self):
        partial_results = self.text[:-2]  # Removing last '\n'
        self.text = ''
        return partial_results


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_accuracy(predictions, labels):
    cf = confusion_matrix(labels, predictions).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = [h/c if c > 0 else 0.00 for (h, c) in zip(cls_hit, cls_cnt)]

    return np.mean(cls_acc) * 100
