import csv
import os

import torch

from .video_dataset import VideoDataset, VideoRecord


class Charades(VideoDataset):
    """ Charades Dataset """
    num_classes = 157
    multi_label = True

    def _parse_list(self):
        """
        Parses the annotation file to create a list of the videos relative path and their labels
        in the format: [label, video_path].
        """
        video_list = []
        with open(self.list_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['id'] + '.mp4'
                actions = row['actions']
                if actions == '':
                    actions = []
                else:
                    actions = [a.split(' ') for a in actions.split(';')]
                    actions = [{
                        'class': x,
                        'start': float(y),
                        'end': float(z)
                    } for x, y, z in actions]
                video_list.append([actions, vid])

        if self.subset:  # Subset for tests!!!
            video_list = [v for i, v in enumerate(video_list) if i % 10 == 0]

        self.video_list = video_list

    def _get_train_target(self, record, offsets):
        """
        Args:
            record : VideoRecord object
            offsets : List of image indices to be loaded from a video.
        Returns:
            target: Dict with the binary list of labels from a video.
        """
        target = torch.IntTensor(self.num_classes).zero_()
        for frame in offsets:
            for l in record.label:
                if l['start'] < frame / float(record.fps) < l['end']:
                    target[int(l['class'][1:])] = 1

        return {'target': target}

    def _get_test_target(self, record):
        """
        Args:
            record : VideoRecord object
        Returns:
            target: Dict with the binary list of labels from a video and its relative path.
        """
        target = torch.IntTensor(self.num_classes).zero_()
        for l in record.label:
            target[int(l['class'][1:])] = 1

        return {'target': target, 'video_path': os.path.splitext(os.path.basename(record.path))[0]}


def calculate_charades_pos_weight(list_file, root_path, output_dir):
    video_list = []
    total_frames = 0
    with open(list_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']

            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'class': x, 'start': float(y), 'end': float(z)} for x, y, z in actions]

            record = VideoRecord(os.path.join(root_path, vid + '.mp4'), actions)
            total_frames += int(float(row['length']) * record.fps)

            video_list.append([actions, record.fps])

    positive_frames_per_class = torch.FloatTensor(157).zero_()
    for label, fps in video_list:
        for l in label:
            frame_start = int(l['start'] * fps)
            frame_end = int(l['end'] * fps)
            positive_frames_per_class[int(l['class'][1:])] += frame_end - frame_start
    pos_weight = torch.FloatTensor([(total_frames - p) / p for p in positive_frames_per_class])

    torch.save(pos_weight, os.path.join(output_dir, 'charades_pos_weight.pt'))
