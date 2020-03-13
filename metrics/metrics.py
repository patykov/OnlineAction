import horovod.torch as hvd
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from .charades_classify import charades_map


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, synchronize=True):
        self.name = name
        self.synchronize = synchronize
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    @torch.no_grad()
    def update(self, val, n=1):
        self.val = val
        if self.synchronize:
            self.sum += hvd.allreduce(torch.tensor(val), average=False,
                                      name=self.name + '_sum').item()
            self.count += hvd.allreduce(torch.tensor(n), average=False,
                                        name=self.name + '_count').item()
        else:
            self.sum += val
            self.count += n
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
    def __init__(self, k=(1, ), video=False):
        self.k = k
        self.maxk = max(k)
        self.video = video
        self.softmax2d = torch.nn.Softmax2d()
        super().__init__('/'.join(['top{}'.format(ki) for ki in k]))

    def reset(self):
        super().reset()
        self.labels = None
        self.total = 0
        self.topk_hit = [0 for i in self.k]

    def _add(self, output, target, synchronize=True):
        print(output.shape, target.shape)
        if output.ndim > 2:
            output = self.softmax2d(output)
            output = output.mean(2).mean(2)
        else:
            output = output.softmax(dim=1)
        print(output.shape, target.shape)

        if self.video:
            output = output.mean(0)

        _, topk_labels = torch.topk(output, self.maxk)

        print(target.shape, torch.stack([target.cpu()], dim=1).shape)

        if synchronize:
            self.targets = hvd.allgather(
                target.unsqueeze(1).cpu(), name=self.name + '_target')
            self.labels = hvd.allgather(topk_labels.cpu(), name=self.name + '_label')
            self.predictions = hvd.allgather(output.cpu(), name=self.name + '_pred')
        else:
            self.targets = target.unsqueeze(1).cpu()
            self.labels = topk_labels.cpu()
            self.predictions = output.cpu()

        self.total += self.targets.shape[0]
        print(self.labels.shape)
        for i, k in enumerate(self.k):
            self.topk_hit[i] += torch.sum(self.targets == self.labels[:, :k])

    def _get_value(self):
        return self.topk_hit / self.total.astype(float)

    def __repr__(self):
        return '/'.join(['{:.02%}'.format(v) for v in [*self.value]])


class PerClassTopK(Metric):
    def __init__(self, k=(1, ), video=False):
        super().__init__('/'.join(['top{}'.format(ki) for ki in k]))
        self.k = k
        self.maxk = max(k)
        self.video = video
        self.softmax2d = torch.nn.Softmax2d()

    def reset(self):
        super().reset()
        self.labels = []

    def _add(self, output, target, synchronize=True):
        if output.ndim > 2:
            output = self.softmax2d(output)
            output = output.mean(2).mean(2)
        else:
            output = output.softmax(dim=1)

        if self.video:
            output = output.mean(0)

        _, topk_labels = torch.topk(output, self.maxk)

        if synchronize:
            self.targets.append(
                hvd.allgather(torch.stack([target.cpu()], dim=1), name=self.name + '_target'))
            self.labels.append(hvd.allgather(topk_labels.cpu(), name=self.name + '_label'))
            self.predictions.append(hvd.allgather(output.cpu(), name=self.name + '_pred'))
        else:
            self.targets.append(torch.stack([target.cpu()], dim=1))
            self.labels.append(topk_labels.cpu())
            self.predictions.append(output.cpu())

    def _get_value(self):
        labels = np.vstack(self.labels)
        targets = np.vstack(self.targets)
        acc = []
        for ki in self.k:
            label_ki = [t.item() if t in p[:ki] else p[0] for p, t in zip(labels, targets)]
            acc.append(per_class_accuracy(label_ki, targets))
        return acc

    def __repr__(self):
        return '/'.join(['{:.02%}'.format(v) for v in [*self.value]])


class mAP(Metric):
    def __init__(self, video=False):
        super().__init__('mAP')
        self.video = video

    def _add(self, output, target, synchronize=True):
        output = torch.sigmoid(output)

        if output.ndim > 2:
            output = output.mean(2).mean(2)

        if self.video:
            output = output.max(0)

        if synchronize:
            self.targets.append(hvd.allgather(target.cpu(), name=self.name + '_target'))
            self.predictions.append(hvd.allgather(output.cpu(), name=self.name + '_pred'))
        else:
            self.targets.append(target.cpu())
            self.predictions.append(output.cpu())

    def _get_value(self):
        mAP, _, _, _, _, _ = charades_map(np.vstack(self.predictions), np.vstack(self.targets))
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
        self.metric.add(output, target['target'], **kargs)
        self.update_text(target)

    def update_text(self, target):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.metric)

    def to_text(self):
        partial_results = self.text.strip()  # Removing last '\n'
        self.text = ''
        return partial_results


class VideoPerFrameAccuracy(VideoWrapper):

    def update_text(self, target):
        batch_size = target['target'].shape[0]
        for img_id in range(batch_size):
            self.text += '{} | {} | {} \n'.format(
                target['video_path'][img_id],
                target['target'][img_id].item(),
                np.array2string(self.metric.predictions[img_id].numpy(),
                                separator=' ',
                                precision=4,
                                suppress_small=True)[1:-1])


class VideoPerFramePerClassAccuracy(VideoWrapper):

    def update_text(self, target):
        batch_size = target['target'].shape[0]
        for img_id in range(batch_size):
            self.text += '{} | {} | {} \n'.format(
                target['video_path'][img_id],
                target['target'][img_id].item(),
                np.array2string(self.metric.predictions[-1][img_id].numpy(),
                                separator=' ',
                                precision=4,
                                suppress_small=True)[1:-1])


class VideoPerFrameMAP(VideoWrapper):
    def update_text(self, target):
        batch_size = target['target'].shape[0]
        for img_id in range(batch_size):
            self.text += '{} {} \n'.format(
                target['video_path'][img_id],
                np.array2string(self.metric.predictions[-1][img_id].numpy(),
                                separator=' ',
                                precision=8)[1:-1])


class VideoMAP(VideoWrapper):
    def update_text(self, target):
        self.text += '{} {} \n'.format(
            target['video_path'],
            np.array2string(self.metric.predictions[-1].numpy(),
                            separator=' ',
                            precision=8)[1:-1])


class VideoAccuracy(VideoWrapper):
    def update_text(self, target):
        self.text += '{:^5} | {:^20} \n'.format(
            target['target'][0],
            np.array2string(self.metric.labels.numpy(),
                            separator=' ')[1:-1])


class VideoPerClassAccuracy(VideoWrapper):
    def update_text(self, target):
        self.text += '{:^5} | {:^20} \n'.format(
            target['target'][0],
            np.array2string(self.metric.labels[-1].numpy(),
                            separator=' ')[1:-1])


def per_class_accuracy(predictions, labels):
    cf = confusion_matrix(labels, predictions).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    return np.nanmean(cls_hit / cls_cnt)
