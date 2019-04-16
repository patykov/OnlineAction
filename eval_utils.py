
import os
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

# class eval_13d_nl(object):
#     def __init__(self, model, num_depth=32, num_channel=3):
#         self.num_channel = num_channel
#         self.num_depth = num_depth
#         self.model = model

#     def __call__(self, data):
#         data = data.to(device)
#         data = data.squeeze(0)
#         data = data.view(
#             self.num_channel, -1, self.num_depth, data.size(2), data.size(3)).contiguous()
#         data = data.permute(1, 0, 2, 3, 4).contiguous()

#         return self.model(data).mean(0).cpu()


def evaluate_model(i3d_model, eval_video, data_gen, total_num, output_file=None):
    if output_file is not None:
        with open(output_file, 'w') as file:
            file.write('{:^10} | {:^20}\n'.format('Label', 'Top5 predition'))

    proc_start_time = time.time()

    score_text = ''
    video_pred = []
    video_labels = []
    with torch.no_grad():
        for i, (data, label) in data_gen:
            rst = eval_video(data)
            cnt_time = time.time() - proc_start_time

            _, top5_pred = torch.topk(rst, 5)
            video_pred.append(top5_pred)
            video_labels.append(label[0])
            score_text += '{:^10} | {:^20}\n'.format(label[0], np.array2string(
                top5_pred.numpy(), separator=', ')[1:-1])

            if i % 10 == 0:
                print('video {}/{} done, {:.02f}%, average {:.5f} sec/video'.format(
                    i, total_num, i*100/total_num, float(cnt_time)/i))
                if i % 100 == 0:
                    # Saving as the program goes in case of error
                    if output_file is not None:
                        with open(output_file, 'a') as file:
                            file.write(score_text)
                    score_text = ''

    # Saving last < 100 lines
    if output_file is not None:
        with open(output_file, 'a') as file:
            file.write(score_text)

    save_metrics(video_pred, video_labels, output_file)


def get_acc_report(predictions, labels):
    cf = confusion_matrix(labels, predictions).astype(float)

    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = [h/c if c > 0 else 0.00 for (h, c) in zip(cls_hit, cls_cnt)]

    report = classification_report(labels, predictions)

    return np.mean(cls_acc) * 100, report


def save_metrics(video_pred, video_labels, output_file):
    # TOP1
    top1_pred = [p[0] for p in video_pred]
    top1_pred = [p[0].item() for p in video_pred]
    cls_acc1, report1 = get_acc_report(top1_pred, video_labels)

    # TOP 5
    top5_pred = []
    for l, p in zip(video_labels, video_pred):
        if l in p:
            top5_pred.append(l)
        else:
            top5_pred.append(p[0])
    cls_acc5, report5 = get_acc_report(top5_pred, video_labels)

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
