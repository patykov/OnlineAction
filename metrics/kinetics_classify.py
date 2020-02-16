import re

import numpy as np

from metrics.metrics import per_class_accuracy
from sklearn.metrics import classification_report


def get_top_predictions(video_pred, video_labels):
    # TOP1
    top1_pred = [p[0] for p in video_pred]

    # TOP 5
    top5_pred = []
    for l, p in zip(video_labels, video_pred):
        if l in p:
            top5_pred.append(l)
        else:
            top5_pred.append(p[0])

    return top1_pred, top5_pred


def save(file_path, output_file, percentage=False, batch_time=None, data_time=None):
    video_labels, video_pred = read_file(file_path)
    stacked_pred = np.vstack(video_pred)
    stacked_labels = np.hstack(video_labels)

    top1_pred, top5_pred = get_top_predictions(stacked_pred, stacked_labels)

    cls_acc1 = per_class_accuracy(top1_pred, stacked_labels)
    report1 = classification_report(stacked_labels, top1_pred)

    cls_acc5 = per_class_accuracy(top5_pred, stacked_labels)
    report5 = classification_report(stacked_labels, top5_pred)

    print('\n\nAccuracy:\nTop1: {:.02%} | Top5: {:.02%}'.format(cls_acc1, cls_acc5))

    if percentage:
        p_clip_labels = np.array([np.array_split(cl, 10) for cl in video_labels])
        p_clip_pred = np.array([np.array_split(cp, 10) for cp in video_pred])

        percentage_labels = np.array([np.hstack(p_clip_labels[:, i]) for i in range(10)])
        percentage_pred = np.array([np.vstack(p_clip_pred[:, i]) for i in range(10)])

        # Save per 10% accuracy
        p_accuracy = []
        for p_labels, p_preds in zip(percentage_labels, percentage_pred):
            top1_pred, top5_pred = get_top_predictions(p_preds, p_labels)

            cls_acc1 = per_class_accuracy(top1_pred, p_labels)
            cls_acc5 = per_class_accuracy(top5_pred, p_labels)
            p_accuracy.append([cls_acc1, cls_acc5])

    with open(output_file, 'w') as file:
        file.write('### Accuracy ### \n')
        file.write('Top1: {:.02%} | Top5: {:.02%}'.format(cls_acc1, cls_acc5))

        if batch_time and data_time:
            file.write('\n\n### Eval Time ### \n')
            file.write('Batch Time: {batch_time.avg:.3f}s avg. | '
                       'Data loading time: {data_time.avg:.3f}s avg.'.format(
                        batch_time=batch_time, data_time=data_time))

        if percentage:
            file.write('\n\n### % Accuracy ###\n')
            for i, [t1, t5] in enumerate(p_accuracy):
                file.write('{:^3} - {:^3} of video | T1: {:.02%} | T5: {:.02%}\n'.format(
                    i*10, (i+1)*10, t1, t5))

        file.write('\n\n-----------------------------------------------------\n')
        file.write('### Per-class Report ### \n')
        file.write('Top1 report:\n')
        file.write(report1)
        file.write('\n\n-----------------------------------------------------\n')
        file.write('Top5 report:\n')
        file.write(report5)


def read_file(file_path):
    with open(file_path, 'r') as file:
        _ = file.readline()  # removing header
        text = sorted(file.readlines())

    split_text = [t.strip().split('|') for t in text]

    if len(split_text[0]) > 2:
        labels, preds = divide_per_clip(split_text)
    else:
        labels = [int(st[0]) for st in split_text]
        preds = [np.fromstring(st[1], dtype=int, sep=', ') for st in split_text]

    return labels, preds


def divide_per_clip(split_text):
    clips_labels = []
    clips_prediction = []
    video_labels = []
    video_prediction = []

    video_name = None
    for i, [video_path, label, pred] in enumerate(split_text):
        video = re.sub('()_\\d{6}', '', video_path.strip())
        if video != video_name or i == len(split_text):
            # new video! But first, save old video
            if i > 0:
                video_labels = np.array(video_labels)
                video_prediction = np.array(video_prediction)

                num_frames = video_labels.shape[0]
                if num_frames < 10:
                    repeat_ids = np.round(np.linspace(0, num_frames - 1, 10)).astype(int)
                    video_labels = video_labels[repeat_ids]
                    video_prediction = video_prediction[repeat_ids]

                clips_labels.append(video_labels)
                clips_prediction.append(video_prediction)

            # start new one
            video_labels = []
            video_prediction = []
            video_name = video

        video_labels.append(int(label))
        video_prediction.append(np.fromstring(pred, dtype=int, sep=' '))

    return clips_labels, clips_prediction
