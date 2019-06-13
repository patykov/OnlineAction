import csv
import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from numpy.random import randint

import transforms as t

from .video_dataset import VideoRecord


class Charades(data.Dataset):
    """ Charades Dataset.
    Args:
        root_path: Full path to the dataset videos directory.
        list_file: Full path to the file that lists the videos to be considered (train, val, test)
            with its annotations.
        sample_frames: Number of frames used in the input (temporal dim)
        stride: Temporal stride used to collect the sample_frames(E.g.: An input of 32 frames with
            stride of 2 reaches a temporal depth of 64 frames (32x2). As does an input of 8 frames
            with a stride of 8 (8x8).)
        mode: Set the dataset mode as 'train', 'val' or 'test'.
        transform: A function that takes in an PIL image and returns a transformed version.
    """
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    FPS, GAP, testGAP = 24, 4, 25
    num_classes = 157

    def __init__(self, root_path, list_file, sample_frames=32, transform=None,
                 mode='train', test_clips=10):
        self.root_path = root_path
        self.sample_frames = sample_frames
        self.stride = 2 if self.sample_frames == 32 else 8
        self.mode = mode
        self.test_clips = test_clips

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()

        self.video_list = self._parse_list(list_file)

    def _parse_list(self, list_file):
        """
        Argument:
            list_file : File that contains each video relative path and its annotation
        Returns:
            video_list: List of the videos relative path and their labels in the format:
                        [label, video_path].
        """
        video_list = []
        with open(list_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                vid = row['id']
                actions = row['actions']
                if actions == '':
                    actions = []
                else:
                    actions = [a.split(' ') for a in actions.split(';')]
                    actions = [{'class': x, 'start': float(
                        y), 'end': float(z)} for x, y, z in actions]
                video_list.append([actions, vid])

        return video_list

    def _get_train_indices(self, record):
        expanded_sample_length = self.sample_frames * self.stride
        if record.num_frames >= expanded_sample_length:
            start_pos = randint(record.num_frames - expanded_sample_length + 1)
            offsets = range(start_pos, start_pos + expanded_sample_length, self.stride)
        elif record.num_frames > self.sample_frames:
            start_pos = randint(record.num_frames - self.sample_frames + 1)
            offsets = range(start_pos, start_pos + self.sample_frames, 1)
        else:
            offsets = np.sort(randint(record.num_frames, size=self.sample_frames))

        offsets = [int(v) for v in offsets]

        target = torch.IntTensor(157).zero_()
        for frame in offsets:
            for l in record.label:
                if l['start'] < frame/float(self.FPS) < l['end']:
                    target[int(l['class'][1:])] = 1
        return offsets, target

    def _get_test_indices(self, record):
        """
        Argument:
            record : VideoRecord object
        Returns:
            offsets : List of image indices to be loaded
            targets: List of
        """
        tick = (record.num_frames - self.sample_frames*self.stride + 1) / float(self.test_clips)
        sample_start_pos = np.array([int(tick * x) for x in range(self.test_clips)])
        offsets = []
        for p in sample_start_pos:
            offsets.extend(range(p, p+self.sample_frames*self.stride, self.stride))
        print(record.num_frames, self.sample_frames, self.stride, tick, sample_start_pos)
        checked_offsets = []
        for f in offsets:
            new_f = int(f)
            if new_f < 0:
                new_f = 0
            elif new_f >= record.num_frames:
                new_f = record.num_frames - 1
            checked_offsets.append(new_f)

        target = torch.IntTensor(157).zero_()
        for l in record.label:
            target[int(l['class'][1:])] = 1

        return checked_offsets, target

    def __getitem__(self, index):
        label, video_path = self.video_list[index]
        record = VideoRecord(os.path.join(self.root_path, video_path+'.mp4'), label)

        if self.mode == 'train':
            segment_indices, target = self._get_train_indices(record)
            process_data = self.get(record, segment_indices)
            while process_data is None:
                index = randint(0, len(self.video_list) - 1)
                process_data, target = self.__getitem__(index)
        else:
            segment_indices, target = self._get_test_indices(record)
            process_data = self.get(record, segment_indices)
            if process_data is None:
                raise ValueError('sample indices:', record.path, segment_indices)

        data = process_data.squeeze(0)
        data = data.view(3, -1, self.sample_frames, data.size(2), data.size(3)).contiguous()
        data = data.permute(1, 0, 2, 3, 4).contiguous()

        return data, target

    def get(self, record, indices):
        uniq_id = np.unique(indices)
        uniq_imgs = record.get_frames(uniq_id)

        if None in uniq_imgs:
            return None

        images = [uniq_imgs[i] for i in indices]
        images = self.transform(images)
        return images

    def __len__(self):
        return len(self.video_list)

    def default_transforms(self):
        """
        Returns:
            A transform function to be applied in the PIL images.
        """
        if self.mode == 'val':
            cropping = torchvision.transforms.Compose([
                t.GroupResize(256),
                t.GroupCenterCrop(224)
            ])
        elif self.mode == 'test':
            cropping = torchvision.transforms.Compose([
                t.GroupResize(256),
                t.GroupFullyConv(256)
            ])
        elif self.mode == 'train':
            cropping = torchvision.transforms.Compose([
                t.GroupRandomResize(256, 320),
                t.GroupRandomCrop(224),
                t.GroupRandomHorizontalFlip()
            ])
        else:
            raise ValueError('Mode {} does not exist. Choose between: val, test or train.'.format(
                self.mode))

        transforms = torchvision.transforms.Compose([
                cropping,
                t.GroupToTensorStack(),
                t.GroupNormalize(mean=self.input_mean, std=self.input_std)
            ])

        return transforms
