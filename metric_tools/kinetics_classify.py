import numpy as np
from sklearn.metrics import classification_report

from metric_tools.metrics import get_accuracy


def get_top_predictions(video_pred, video_labels):
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


def save(video_pred, video_labels, output_file, batch_time=None, data_time=None):
    top1_pred, top5_pred = get_top_predictions(video_pred, video_labels)

    cls_acc1 = get_accuracy(top1_pred, video_labels)
    report1 = classification_report(video_labels, top1_pred)

    cls_acc5 = get_accuracy(top5_pred, video_labels)
    report5 = classification_report(video_labels, top5_pred)

    print('\n\nAccuracy:\nTop1: {:.02f}% | Top5: {:.02f}%'.format(cls_acc1, cls_acc5))

    with open(output_file, 'w') as file:
        file.write('### Accuracy ### \n')
        file.write('Top1: {:.02f}% | Top5: {:.02f}%'.format(cls_acc1, cls_acc5))
        if batch_time and data_time:
            file.write('\n\n### Eval Time ### \n')
            file.write('Batch Time: {batch_time.avg:.3f}s avg. | '
                       'Data loading time: {data_time.avg:.3f}s avg.'.format(
                        batch_time=batch_time, data_time=data_time))
        file.write('\n\n-----------------------------------------------------\n')
        file.write('### Per-class Report ### \n')
        file.write('Top1 report:\n')
        file.write(report1)
        file.write('\n\n-----------------------------------------------------\n')
        file.write('Top5 report:\n')
        file.write(report5)


def save_causal(video_pred, video_labels, output_file, batch_time=None, data_time=None):
    cls_acc1 = []
    cls_acc5 = []
    for i in range(10):
        v_pred = [p[i] for p in video_pred]
        top1_pred, top5_pred = get_top_predictions(v_pred, video_labels)

        acc1 = get_accuracy(top1_pred, video_labels)
        acc5 = get_accuracy(top5_pred, video_labels)
        cls_acc1.append(acc1)
        cls_acc5.append(acc5)

        print('Accuracy for {:3}%: Top1: {:.02f}% | Top5: {:.02f}%'.format(i*10, acc1, acc5))

    with open(output_file, 'w') as file:
        file.write('### Accuracy ### \n')
        file.write('{:10} | {:6} | {:6}\n'.format('% of video', 'Top1', 'Top5'))
        for i in range(10):
            file.write('{:9}% | {:5.02f}% | {:5.02f}%\n'.format(
                i*10, cls_acc1[i], cls_acc5[i]))

        if batch_time and data_time:
            file.write('\n\n### Eval Time ###\n')
            file.write('Batch Time: {batch_time.avg:.3f}s avg. | '
                       'Data loading time: {data_time.avg:.3f}s avg.\n'.format(
                        batch_time=batch_time, data_time=data_time))


def read_file(file_path):
    with open(file_path, 'r') as file:
        header = file.readline()  # removing header
        text = file.readlines()

    split_text = [t.replace('\n', '').split('|') for t in text]
    labels = [int(st[0]) for st in split_text]

    if len(header.split('|')) > 2:
        is_causal = True
        preds = [[np.fromstring(s, dtype=int, sep=', ') for s in st[1:]] for st in split_text]

    else:
        is_causal = False
        preds = [np.fromstring(st[1], dtype=int, sep=', ') for st in split_text]

    return labels, preds, is_causal
