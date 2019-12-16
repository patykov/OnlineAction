import csv
import os
import re

import numpy as np
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
                        'class': int(x[1:]),
                        'start': float(y),
                        'end': float(z)
                    } for x, y, z in actions]
                video_list.append([actions, vid])

        if self.subset:  # Subset for tests!!!
            video_list = [v for i, v in enumerate(video_list) if i % 10 == 0]

        self.video_list = video_list

    def select_classes_over_thres(self, file_path):
        if file_path:
            selected_classes = {}
            with open(file_path, 'r') as file:
                for line in file:
                    class_match = re.match('(\d*) (.*) (\d*)', line)
                    if class_match:
                        new_class_id, old_class_id, class_name = class_match.groups()
                    selected_classes[int(old_class_id)] = int(new_class_id)

            # Set new num_classes
            self.num_classes = len(selected_classes)

            # Select classes from gt
            new_video_list = []
            for actions, vid in self.video_list:
                new_actions = []
                for a in actions:
                    if a['class'] in selected_classes:
                        old_value = a['class']
                        a['class'] = selected_classes[old_value]
                        new_actions.append(a)
                new_video_list.append([new_actions, vid])

            self.video_list = new_video_list

    def select_classes_by_verb(self, file_path):
        if file_path:
            class_to_verb = {}
            with open(file_path, 'r') as file:
                for line in file:
                    class_match = re.match('(\d*) (.*) (\d*)', line)
                    if class_match:
                        old_class_id, new_class_id, new_class_name = class_match.groups()
                    class_to_verb[int(old_class_id)] = int(new_class_id)

            # Set new num_classes
            self.num_classes = len(np.unique(list(class_to_verb.values())))

            # Select classes from gt
            new_video_list = []
            for actions, vid in self.video_list:
                new_actions = []
                for a in actions:
                    if a['class'] in class_to_verb:
                        old_value = a['class']
                        a['class'] = class_to_verb[old_value]
                        new_actions.append(a)
                new_video_list.append([new_actions, vid])

            self.video_list = new_video_list

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
                    target[l['class']] = 1

        return {'target': target, 'video_path': os.path.splitext(os.path.basename(record.path))[0]}

    def _get_test_target(self, record):
        """
        Args:
            record : VideoRecord object
        Returns:
            target: Dict with the binary list of labels from a video and its relative path.
        """
        target = torch.IntTensor(self.num_classes).zero_()
        for l in record.label:
            target[l['class']] = 1

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
