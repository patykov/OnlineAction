import os

import numpy as np
import torch.utils.data as data
import torchvision
from numpy.random import randint

import transforms as t

from .video_record import VideoRecord


class VideoDataset(data.Dataset):
    """ A generic video dataset.
    Args:
        root_path: Full path to the dataset videos directory.
        list_file: Full path to the file that lists the videos to be considered (train, val, test)
            with its annotations.
        sample_frames: Number of frames used in the input (temporal dimension).
        transform: A function that takes in an PIL image and returns a transformed version.
        mode: Set the dataset mode as 'train', 'val' or 'test'.
        test_clips: Number of clips to be evenly sampled from each full-length video for evaluation.
    """
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    num_classes = None
    multi_label = None

    def __init__(self, root_path, list_file, sample_frames=32, transform=None,
                 mode='train', test_clips=10, subset=False):
        self.root_path = root_path
        self.list_file = list_file
        self.sample_frames = sample_frames
        self.clip_length = 3  # in seconds
        self.mode = mode
        self.test_clips = test_clips
        self.subset = subset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()

        self._parse_list()

    def _parse_list(self):
        """
        Parses the annotation file to create a list of the videos relative path and their labels
        in the format: [label, video_path].
        """
        self.video_list = None
        raise NotImplementedError()

    def _get_train_indices(self, record):
        """
        Args:
            record : VideoRecord object
        Returns:
            offsets : List of image indices to be loaded from a video.
        """
        expanded_sample_length = int(self.clip_length * record.fps)
        if record.num_frames > expanded_sample_length:
            start_pos = randint(record.num_frames - expanded_sample_length)
            offsets = np.linspace(
                start_pos, start_pos + expanded_sample_length, self.sample_frames, dtype=int)
        elif record.num_frames > self.sample_frames:
            start_pos = randint(record.num_frames - self.sample_frames)
            offsets = np.linspace(
                start_pos, start_pos + self.sample_frames, self.sample_frames, dtype=int)
        else:
            offsets = np.linspace(0, record.num_frames - 1, self.sample_frames, dtype=int)

        return offsets

    def _get_test_indices(self, record):
        """
        Args:
            record : VideoRecord object
        Returns:
            offsets : List of image indices to be loaded from a video.
        """
        expanded_sample_length = int(self.clip_length * record.fps)
        sample_start_pos = np.linspace(
            expanded_sample_length, record.num_frames-1, self.test_clips, dtype=int)
        offsets = []
        for p in sample_start_pos:
            offsets.extend(np.linspace(
                max(p-expanded_sample_length, 0),
                min(p, record.num_frames-1),
                self.sample_frames, dtype=int))

        return offsets

    def _get_train_target(self, record, *args):
        raise NotImplementedError()

    def _get_test_target(self, record):
        raise NotImplementedError()

    def __getitem__(self, index):
        label, video_path = self.video_list[index]
        try:
            record = VideoRecord(os.path.join(self.root_path, video_path), label)

            if self.mode in ['train', 'val']:
                segment_indices = self._get_train_indices(record)
                target = self._get_train_target(record, segment_indices)
            else:
                segment_indices = self._get_test_indices(record)
                target = self._get_test_target(record)
            data = self.get(record, segment_indices)

        except ValueError as e:
            print(e)
            index = randint(0, len(self.video_list) - 1)
            data, target = self.__getitem__(index)

        return data, target

    def get(self, record, indices):
        """
        Args:
            record : VideoRecord object
            indices : List of image indices to be loaded from a video.
        Returns:
            data : Numpy array with the loaded and transformed data from the video.
        """
        uniq_id = np.unique(indices)
        uniq_imgs = record.get_frames(uniq_id)

        images = [uniq_imgs[i] for i in indices]
        images = self.transform(images)

        data = images.view(3, -1, self.sample_frames, images.size(2), images.size(3))
        data = data.permute(1, 0, 2, 3, 4)
        return data

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

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of samples: {}\n'.format(self.__len__())
        fmt_str += '    Videos Location: {}\n'.format(self.root_path)
        fmt_str += '    Annotations file: {}\n'.format(self.list_file)
        fmt_str += '    Test clips: {}\n'.format(self.test_clips) if self.mode == 'test' else ''
        tmp = ' (multi-label)' if self.multi_label else ''
        fmt_str += '    Number of classes: {}{}\n'.format(self.num_classes, tmp)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str
