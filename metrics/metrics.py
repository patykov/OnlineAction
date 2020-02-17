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

    def _add(self, output, target, synchronize=True):
        _, topk_pred = torch.topk(output, self.maxk)

        if synchronize:
            self.targets.append(
                hvd.allgather(torch.stack([target.cpu()], dim=1), name=self.name + '_target'))
            self.predictions.append(hvd.allgather(topk_pred.cpu(), name=self.name + '_pred'))
        else:
            self.targets.append(torch.stack([target.cpu()], dim=1))
            self.predictions.append(topk_pred.cpu())

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

    def _add(self, output, target, synchronize=True):
        prediction = torch.sigmoid(output)

        if synchronize:
            self.targets.append(hvd.allgather(target.cpu(), name=self.name + '_target'))
            self.predictions.append(hvd.allgather(prediction.cpu(), name=self.name + '_pred'))
        else:
            self.targets.append(target.cpu())
            self.predictions.append(prediction.cpu())

    def _get_value(self):
        mAP, _, _, _, _ = charades_map(np.vstack(self.predictions), np.vstack(self.targets))
        return mAP

    def __repr__(self):
        return '{:.02%}'.format(self.value)


class VideoWrapper:
    """ Metric wrapper that stores model predictions at video level evaluation in a text
    format."""
    def __init__(self, metric):
        self.metric = metric
        self.reset()

    def reset(self):
        self.text = ''
        self.metric.reset()

    def add(self, output, target, **kargs):
        self._add(output, target, **kargs)

    def _add(self, output, target, **kargs):
        raise NotImplementedError()

    def update_text(self, target):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.metric)

    def to_text(self):
        partial_results = self.text.strip()  # Removing last '\n'
        self.text = ''
        return partial_results


class VideoPerFrameAccuracy(VideoWrapper):
    def __init__(self, metric):
        super().__init__(metric)
        self.video_target = None
        self.video_predictions = []

    def reset(self):
        self.text = '{:^40} | {:^5} | {}\n'.format(
            'Path', 'Label', 'Top5 predition')
        self.metric.reset()

    def update_text(self, target):
        batch_size = target['target'].shape[0]
        for img_id in range(batch_size):
            pred = self.metric.predictions[-1][img_id].numpy()
            assert pred.shape == (5,)  # REMOVE LATER
            self.text += '{:40} | {:^5} | {}\n'.format(
                target['video_path'][img_id][0],
                target['target'][img_id].item(),
                np.array2string(pred, separator=' ')[1:-1])

    def _add(self, output, target, synchronize=True):
        if synchronize:
            self.targets.append(hvd.allgather(target.cpu(), name=self.name + '_target'))
            self.predictions.append(hvd.allgather(prediction.cpu(), name=self.name + '_pred'))
        else:
            self.targets.append(target.cpu())
            self.predictions.append(prediction.cpu())

    # def get_video_prediction(self):
        # percentage_result = []
        # for i in range(1, 11):



class VideoPerFrameMAP(VideoWrapper):
    def update_text(self, target):
        batch_size = target['target'].shape[0]
        for img_id in range(batch_size):
            self.text += '{} {}\n'.format(target['video_path'][img_id], np.array2string(
                self.metric.predictions[-1][img_id].numpy(), separator=' ',
                formatter={'float_kind': lambda x: '%.8f' % x})[1:-1].replace('\n', ''))

    def _add(self, output, target, **kargs):
        self.metric.add(output, target['target'], **kargs)
        self.update_text(target)


class VideoMAP(VideoWrapper):
    def update_text(self, target):
        self.text += '{} {}\n'.format(target['video_path'][0], np.array2string(
            self.metric.predictions[-1].numpy(), separator=' ',
            formatter={'float_kind': lambda x: '%.8f' % x})[1:-1].replace('\n', ''))

    def _add(self, output, target, **kargs):
        self.metric.add(output, target['target'], **kargs)
        self.update_text(target)


class VideoAccuracy(VideoWrapper):
    def reset(self):
        self.text = '{:^5} | {:^20}\n'.format('Label', 'Top5 predition')
        self.metric.reset()

    def update_text(self, target):
        self.text += '{:^5} | {:^20}\n'.format(target['target'][0], np.array2string(
            self.metric.predictions[-1].numpy(), separator=', ')[1:-1])

    def _add(self, output, target, **kargs):
        self.metric.add(output, target['target'], **kargs)
        self.update_text(target)


def per_class_accuracy(predictions, labels):
    cf = confusion_matrix(labels, predictions).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    return np.nanmean(cls_hit/cls_cnt)
