import os

import numpy as np
import torch

from eval_metrics import get_results_file_name
from metric_tools.kinetics_classify import save, save_causal


class KineticsLog(object):

    default_log_path = os.path.join(
        os.path.dirname(__file__), '..', 'outputs', 'eval_kinetics_log.txt')

    def __init__(self, output_file=None, causal=False, test_clips=10):
        self.output_file = output_file if output_file else KineticsLog.default_log_path
        self.causal = causal
        self.test_clips = test_clips
        self.text = ''
        self.video_pred = []
        self.video_labels = []
        self.__create_file()

    def __create_file(self):
        output_dir = os.path.dirname(self.output_file)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with open(self.output_file, 'w') as file:
            if self.causal:
                file.write('{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | \n'.format(
                    'Label', 'Top5 - 10%', 'Top5 - 20%', 'Top5 - 30%', 'Top5 - 40%', 'Top5 - 50%',
                    'Top5 - 60%', 'Top5 - 70%', 'Top5 - 80%', 'Top5 - 90%', 'Top5 - 100%'))
            else:
                file.write('{:^10} | {:^20}\n'.format('Label', 'Top5 predition'))

    def update(self, results, label):
        if self.causal:
            clip_depth = int(results.shape[0]/self.test_clips)
            rst = [results[:i*clip_depth].shape for i in range(1, self.test_clips+1)]

            top5_pred = []
            top5_str = ''
            for j in range(10):
                _, t5_p = torch.topk(rst[j], 5)
                top5_pred.append(t5_p)
                top5_str += '{} | '.format(np.array2string(t5_p.numpy(), separator=' ')[1:-1])
            self.text += ('{} | {}\n').format(label[0], top5_str)

        else:
            rst = results.mean(0)

            _, top5_pred = torch.topk(rst, 5)
            self.text += '{:^10} | {:^20}\n'.format(label[0], np.array2string(
                top5_pred.numpy(), separator=', ')[1:-1])

        self.video_pred.append(top5_pred)
        self.video_labels.append(label[0])

    def save_partial(self):
        with open(self.output_file, 'a') as file:
            file.write(self.text)
        self.text = ''

    def get_metrics(self, batch_time, data_time):
        if self.text:
            self.save_partial()  # Make sure all results are saved

        results_file = get_results_file_name(self.output_file)
        if self.causal:
            save_causal(
                self.video_pred, self.video_labels, results_file, batch_time, data_time)
        else:
            save(
                self.video_pred, self.video_labels, results_file, batch_time, data_time)
