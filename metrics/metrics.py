import horovod.torch as hvd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .charades_classify import charades_map


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, val, n=1):
        self.val = val
        self.sum += hvd.allreduce(
            torch.tensor(val), average=False, name=self.name + '_sum').item()
        self.count += hvd.allreduce(
            torch.tensor(n), average=False, name=self.name + '_count').item()
        self.avg = self.sum / self.count


class Metric:
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []

    @torch.no_grad()
    def add(self, output, target):
        self._add(output, target)

    @property
    def value(self):
        return self._get_value()

    @property
    def count(self):
        return len(np.vstack(self.targets))

    def _add(self, output, target):
        raise NotImplementedError()

    def _get_value(self):
        raise NotImplementedError


class Accuracy(Metric):
    def __init__(self):
        super().__init__('accuracy')

    def _add(self, output, target):
        _, top_pred = torch.topk(output, 1)

        self.targets.append(
            hvd.allgather(torch.stack([target.cpu()], dim=1), name=self.name + '_target'))
        self.predictions.append(hvd.allgather(top_pred.cpu(), name=self.name + '_top_pred'))

    def _get_value(self):
        return get_accuracy(np.vstack(self.predictions), np.vstack(self.targets))

    def __repr__(self):
        return '{:.02%}'.format(self.value)


class Top5(Metric):
    def __init__(self):
        super().__init__('accuracy (t1/t5)')

    def _add(self, output, target):
        _, top5_pred = torch.topk(output, 5)

        self.targets.append(
            hvd.allgather(torch.stack([target.cpu()], dim=1), name=self.name + '_target'))
        self.predictions.append(hvd.allgather(top5_pred.cpu(), name=self.name + '_top5_pred'))

    def _get_value(self):
        predictions = np.vstack(self.predictions)
        targets = np.vstack(self.targets)
        acc1 = get_accuracy(predictions[:, 0], targets)
        acc5 = get_accuracy([t.item() if t in p else p[0] for p, t in zip(
            predictions, targets)], targets)
        return acc1, acc5

    def __repr__(self):
        return '{:.02%} / {:.02%}'.format(*self.value)


class mAP(Metric):
    def __init__(self):
        super().__init__('mAP')

    def _add(self, output, target):
        prediction = torch.sigmoid(output)

        self.targets.append(hvd.allgather(target.cpu(), name=self.name + '_target'))
        self.predictions.append(hvd.allgather(prediction.cpu(), name=self.name + '_pred'))

    def _get_value(self):
        mAP, _, _ = charades_map(np.vstack(self.predictions), np.vstack(self.targets))
        return mAP

    def __repr__(self):
        return '{:.02%}'.format(self.value)


class Video_Wrapper:
    """ Metric wrapper that stores model predictions at video level evaluation in a text
    format."""
    def __init__(self, metric):
        self.metric = metric()
        self.reset()

    def reset(self):
        self.text = ''
        self.metric.reset()

    def add(self, output, target):
        self.metric.add(output, target['target'])
        self.update_text(target)

    def update_text(self, target):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.metric)

    def to_text(self):
        partial_results = self.text[:-2]  # Removing last '\n'
        self.text = ''
        return partial_results


class Video_mAP(Video_Wrapper):
    def update_text(self, target):
        self.text += '{} {}\n'.format(target['video_path'][0], np.array2string(
            self.metric.predictions[-1].numpy(), separator=' ',
            formatter={'float_kind': lambda x: '%.8f' % x})[1:-1].replace('\n', ''))


class Video_Accuracy(Video_Wrapper):
    def reset(self):
        self.text = '{:^5} | {:^20}\n'.format('Label', 'Top5 predition')
        self.metric.reset()

    def update_text(self, target):
        self.text += '{:^5} | {:^20}\n'.format(target['target'][0], np.array2string(
            self.metric.predictions[-1].numpy(), separator=', ')[1:-1])


def get_accuracy(predictions, labels):
    cf = confusion_matrix(labels, predictions).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = [h/c if c > 0 else 0.00 for (h, c) in zip(cls_hit, cls_cnt)]

    return np.mean(cls_acc)
