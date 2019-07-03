import os

import numpy as np
import torch.utils.data as data
import torchvision
from numpy.random import randint

import transforms as t
from log_tools.kinetics_log import KineticsLog

from .video_dataset import VideoRecord


class Kinetics(data.Dataset):
    """ Kinetics-400 Dataset.
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
        test_clips: Number of clips to be evenly sample from each full-length video for evaluation.
        causal: Bool to set the type of evaluation.
    """
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    num_classes = 400
    multi_label = False

    def __init__(self, root_path, list_file, sample_frames=32, transform=None,
                 mode='train', test_clips=10, causal=False):
        self.root_path = root_path
        self.list_file = list_file
        self.sample_frames = sample_frames
        self.stride = 2 if self.sample_frames == 32 else 8
        self.mode = mode
        self.test_clips = test_clips
        self.causal = causal

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()

        self.video_list = self._parse_list()

    def _parse_list(self):
        """
        Argument:
            list_file : File that contains each video relative path and its annotation
        Returns:
            List of the videos relative path and their labels in the format: [label, video_path].
        """
        return [x.strip().split(' ') for x in open(self.list_file)]

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
        return offsets

    def _get_test_indices(self, record):
        """
        Argument:
            record : VideoRecord object
        Returns:
            offsets : List of image indices to be loaded
        """
        sample_start_pos = np.linspace(
            self.sample_frames*self.stride, record.num_frames-1, self.test_clips, dtype=int)
        offsets = []
        for p in sample_start_pos:
            offsets.extend(np.linspace(
                max(p-self.sample_frames*self.stride + self.stride, 0),
                min(p, record.num_frames-1),
                self.sample_frames, dtype=int))

        return offsets

    def __getitem__(self, index):
        label, video_path = self.video_list[index]
        record = VideoRecord(os.path.join(self.root_path, video_path), label)

        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
            process_data = self.get(record, segment_indices)
            while process_data is None:
                index = randint(0, len(self.video_list) - 1)
                process_data, label = self.__getitem__(index)
        else:
            segment_indices = self._get_test_indices(record)
            process_data = self.get(record, segment_indices)
            if process_data is None:
                raise ValueError('sample indices:', record.path, segment_indices)

        data = process_data.squeeze(0)
        data = data.view(3, -1, self.sample_frames, data.size(2), data.size(3)).contiguous()
        data = data.permute(1, 0, 2, 3, 4).contiguous()

        return data, int(record.label)

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

    def set_log(self, output_file):
        return KineticsLog(output_file, self.causal, self.test_clips)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Videos Location: {}\n'.format(self.root_path)
        fmt_str += '    Annotations file: {}\n'.format(self.list_file)
        tmp = ' (multi-label)' if self.multi_label else ''
        fmt_str += '    Number of classes: {}{}\n'.format(self.num_classes, tmp)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
