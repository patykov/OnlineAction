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
    def add(self, output, target, **kargs):
        self._add(output, target, **kargs)

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


class TopK(Metric):
    def __init__(self, k=(1,)):
        super().__init__('/'.join(['top{}'.format(ki) for ki in k]))
        self.k = k
        self.maxk = max(k)

    def _add(self, output, target):
        _, topk_pred = torch.topk(output, self.maxk)

        self.targets.append(
            hvd.allgather(torch.stack([target.cpu()], dim=1), name=self.name + '_target'))
        self.predictions.append(hvd.allgather(topk_pred.cpu(), name=self.name + '_pred'))

    def _get_value(self):
        predictions = np.vstack(self.predictions)
        targets = np.vstack(self.targets)
        acc = []
        for ki in self.k:
            pred_ki = [t.item() if t in p[:ki] else p[0] for p, t in zip(predictions, targets)]
            acc.append(per_class_accuracy(pred_ki, targets))
        return acc

    def __repr__(self):
        return '/'.join(['{:.02%}'.format(v) for v in [*self.value]])


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


class cAP(Metric):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__('cAP')

    def reset(self):
        super().reset()
        self.pos_count = np.zeros(self.num_classes)
        self.neg_count = np.zeros(self.num_classes)

    def _add(self, output, target, synchronize=False):
        prediction = torch.sigmoid(output)

        targets = hvd.allgather(
            target.cpu(), name=self.name + '_target') if synchronize else target.cpu()
        self.targets.append(targets)
        self.predictions.append(hvd.allgather(
            prediction.cpu(), name=self.name + '_pred') if synchronize else prediction.cpu())

        sum_targets = sum(targets).numpy()
        self.pos_count += sum_targets
        self.neg_count += (np.ones(self.num_classes)*len(targets) - sum_targets)

    def _get_value(self):
        mAP, _, _ = charades_map(
            np.vstack(self.predictions),
            np.vstack(self.targets),
            self.neg_count/self.pos_count)
        return mAP

    def __repr__(self):
        return '{:.02%}'.format(self.value)


class Video_Wrapper:
    """ Metric wrapper that stores model predictions at video level evaluation in a text
    format."""
    def __init__(self, metric):
        self.metric = metric
        self.reset()

    def reset(self):
        self.text = ''
        self.metric.reset()

    def add(self, output, target, **kargs):
        self.metric.add(output, target['target'], **kargs)
        self.update_text(target)

    def update_text(self, target):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.metric)

    def to_text(self):
        partial_results = self.text[:-2]  # Removing last '\n'
        self.text = ''
        return partial_results


class Video_cAP(Video_Wrapper):
    def update_text(self, target):
        batch_size = target['target'].shape[0]
        for img_id in range(batch_size):
            video_frame = '{}_{:06d}'.format(
                target['video_path'], target['last_frame'] + img_id)

            self.text += '{} {}\n'.format(video_frame, np.array2string(
                self.metric.predictions[-1][img_id].numpy(), separator=' ',
                formatter={'float_kind': lambda x: '%.8f' % x})[1:-1].replace('\n', ''))


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


def per_class_accuracy(predictions, labels):
    cf = confusion_matrix(labels, predictions).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    return np.nanmean(cls_hit/cls_cnt)
