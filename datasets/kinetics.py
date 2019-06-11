import numpy as np
import torchvision

import transforms as t

from .video_dataset import VideoDataset


class Kinetics(VideoDataset):
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
    """
    input_mean = [114.75, 114.75, 114.75]  # [0.485, 0.456, 0.406] -> 114.75 / 255
    input_std = [57.375, 57.375, 57.375]  # [0.229, 0.224, 0.225] --> 57.375 / 255

    def __init__(self, root_path, list_file, sample_frames=32, stride=2, transform=None,
                 mode='train', test_clips=10):
        VideoDataset.__init__(self, root_path, list_file, sample_frames=sample_frames,
                              stride=stride, transform=transform, mode=mode)
        self.stride = 2 if self.sample_frames == 32 else 8
        if self.mode in ['val', 'test']:
            self.num_clips = test_clips

    def _parse_list(self, list_file):
        return [x.strip().split(' ') for x in open(list_file)]

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.sample_frames*self.stride + 1) / float(self.num_clips)
        sample_start_pos = np.array([int(tick * x) for x in range(self.num_clips)])
        offsets = []
        for p in sample_start_pos:
            offsets.extend(range(p, p+self.sample_frames*self.stride, self.stride))

        checked_offsets = []
        for f in offsets:
            new_f = int(f)
            if new_f < 0:
                new_f = 0
            elif new_f >= record.num_frames:
                new_f = record.num_frames - 1
            checked_offsets.append(new_f)

        return checked_offsets

    def default_transforms(self):
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
                t.GroupResize(256),
                t.GroupRandomCrop(224)
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

    @property
    def name(self):
        return 'Kinetics-400'
