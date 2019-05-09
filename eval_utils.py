
import os

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_accuracy(predictions, labels):
    cf = confusion_matrix(labels, predictions).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = [h/c if c > 0 else 0.00 for (h, c) in zip(cls_hit, cls_cnt)]

    return np.mean(cls_acc) * 100


def get_topk_predictions(video_pred, video_labels):
    # TOP1
    top1_pred = [p[0] for p in video_pred]
    top1_pred = [p[0].item() for p in video_pred]

    # TOP 5
    top5_pred = []
    for l, p in zip(video_labels, video_pred):
        if l in p:
            top5_pred.append(l)
        else:
            top5_pred.append(p[0])

    return top1_pred, top5_pred


def save_metrics(video_pred, video_labels, output_file):
    top1_pred, top5_pred = get_topk_predictions(video_pred, video_labels)

    cls_acc1 = get_accuracy(top1_pred, video_labels)
    report1 = classification_report(video_labels, top1_pred)

    cls_acc5 = get_accuracy(top5_pred, video_labels)
    report5 = classification_report(video_labels, top5_pred)

    print('\n\nAccuracy:\nTop1: {:.02f}% | Top5: {:.02f}%'.format(cls_acc1, cls_acc5))

    # Save result report file
    base, file_name = os.path.split(output_file)
    name, ext = os.path.splitext(file_name)
    report_file = os.path.join(base, name+'_results'+ext)

    with open(report_file, 'w') as file:
        file.write('Accuracy:\n')
        file.write('Top1: {:.02f}% | Top5: {:.02f}%\n'.format(cls_acc1, cls_acc5))
        file.write('\n\n-----------------------------------------------------\n')
        file.write('Top1 classification report:\n')
        file.write(report1)
        file.write('\n\n-----------------------------------------------------\n')
        file.write('Top5 classification report:\n')
        file.write(report5)


def save_causal_metrics(video_pred, video_labels, output_file):
    cls_acc1 = []
    cls_acc5 = []
    for i in range(10):
        v_pred = [p[i] for p in video_pred]
        top1_pred, top5_pred = get_topk_predictions(v_pred, video_labels)

        acc1 = get_accuracy(top1_pred, video_labels)
        acc5 = get_accuracy(top5_pred, video_labels)
        cls_acc1.append(acc1)
        cls_acc5.append(acc5)

        print('Accuracy for {:3}%: Top1: {:.02f}% | Top5: {:.02f}%'.format(i*10, acc1, acc5))

    # Save result report file
    base, file_name = os.path.split(output_file)
    name, ext = os.path.splitext(file_name)
    report_file = os.path.join(base, name+'_results'+ext)

    with open(report_file, 'w') as file:
        file.write('{:10} | {:10} | {:10}\n'.format('% of video', 'Top1', 'Top5'))
        for i in range(10):
            file.write('{:10}% | {:10.02f}% | {:10.02f}%\n'.format(i*10, cls_acc1[i], cls_acc5[i]))
