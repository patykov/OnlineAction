import os

import numpy as np
import torch

import metrics as m


class kinetics_log(object):

    default_log_path = os.path.join(os.path.dirname(__file__), 'outputs', 'eval_kinetics_log.txt')

    def __init__(self, output_file=None, causal=False):
        if output_file is None:
            output_file = kinetics_log.default_log_path
        self.output_file = output_file
        self.causal = causal
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
                file.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(
                    'Label', 'Top5 - 10%', 'Top5 - 20%', 'Top5 - 30%', 'Top5 - 40%', 'Top5 - 50%',
                    'Top5 - 60%', 'Top5 - 70%', 'Top5 - 80%', 'Top5 - 90%', 'Top5 - 100%'))
            else:
                file.write('{:^10} | {:^20}\n'.format('Label', 'Top5 predition'))

    def update(self, rst, label):
        if self.causal:
            top5_pred = []
            top5_str = ''
            for j in range(10):
                _, t5_p = torch.topk(rst[j], 5)
                top5_pred.append(t5_p)
                top5_str += '{}\t'.format(np.array2string(t5_p.numpy(), separator=' ')[1:-1])
            self.text += ('{}\t{}\n').format(label[0], top5_str)
        else:
            _, top5_pred = torch.topk(rst, 5)
            self.text += '{:^10} | {:^20}\n'.format(label[0], np.array2string(
                top5_pred.numpy(), separator=', ')[1:-1])

        self.video_pred.append(top5_pred)
        self.video_labels.append(label[0])

    def save_partial(self):
        with open(self.output_file, 'a') as file:
            file.write(self.text)
        self.text = ''

    def save_metrics(self, batch_time, data_time):
        if self.text:
            self.save_partial()  # Make sure all results are saved

        if self.causal:
            m.save_causal_metrics(
                self.video_pred, self.video_labels, self.output_file, batch_time, data_time)
        else:
            m.save_metrics(
                self.video_pred, self.video_labels, self.output_file, batch_time, data_time)
