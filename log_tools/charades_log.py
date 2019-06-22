import os

import numpy as np
import torch

from eval_metrics import get_results_file_name
from metric_tools.charades_classify import save


class CharadesLog(object):

    default_log_path = os.path.join(os.path.dirname(__file__), 'outputs', 'eval_charades_log.txt')

    def __init__(self, gt_file, output_file=None, causal=False, test_clips=10):
        if output_file is None:
            output_file = CharadesLog.default_log_path
        self.output_file = output_file
        self.gt_file = gt_file
        self.causal = causal
        self.test_clips = test_clips
        self.text = ''
        self.video_pred = []
        self.video_ids = []
        self.__create_file()

    def __create_file(self):
        output_dir = os.path.dirname(self.output_file)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if self.causal:
            raise NotImplementedError

        else:
            open(self.output_file, 'w').close()

    def update(self, results, video_id):

        if self.causal:
            raise NotImplementedError

        else:
            rst = results.mean(0)
            pred = torch.sigmoid(rst)
            self.text += '{} {}\n'.format(video_id[0], np.array2string(
                pred.numpy(), separator=' ',
                formatter={'float_kind': lambda x: '%.8f' % x})[1:-1].replace('\n', ''))

        self.video_pred.append(pred)
        self.video_ids.append(video_id[0])

    def save_partial(self):
        with open(self.output_file, 'a') as file:
            file.write(self.text)
        self.text = ''

    def get_metrics(self, batch_time, data_time):
        if self.text:
            self.save_partial()  # Make sure all results are saved

        results_file = get_results_file_name(self.output_file)
        if self.causal:
            raise NotImplementedError
        else:
            save(self.output_file, self.gt_file, results_file)
