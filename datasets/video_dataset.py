import os

import cv2
import numpy as np
import torch.utils.data as data
from numpy.random import randint
from PIL import Image


class VideoRecord(object):
    def __init__(self, video_path, label):
        self.path = video_path
        self.video = cv2.VideoCapture(self.path)
        self.num_frames = self._get_num_frames()
        self.label = label

    def _get_num_frames(self):
        count = 0
        success, frame = self.video.read()
        while(success):
            success, frame = self.video.read()
            count += 1
        self.video.set(2, 0)
        return count

    def get_frames(self, indices):
        """
        Argument:
            indices : Sorted list of frames indices
        Returns:
            images : Dictionary in format {frame_id: PIL Image}
        """
        images = dict()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, min(indices))
        for count in range(min(indices), max(indices)+1):
            success, frame = self.video.read()
            if success is False:
                print('\nCould not load frame {} from video {}\n'.format(count, self.path))
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if count in indices:
                images[count] = Image.fromarray(frame)

        return images


class VideoDataset(data.Dataset):
    """ Base VideoDataset class.
    Argument:
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

    def __init__(self, root_path, list_file, sample_frames=32, stride=2, transform=None,
                 mode='train'):
        self.root_path = root_path
        self.sample_frames = sample_frames
        self.stride = stride
        self.mode = mode

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()

        self.video_list = self._parse_list(list_file)

    def _parse_list(self, list_file):
        """
            Implemented by each child dataset.
        Argument:
            list_file : File that contains each video relative path and its annotation
        Returns:
            video_list : List of the videos relative path and their labels in the format:
                        (label, video_path).
        """
        return None

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

        return data, label

    def get(self, record, indices):
        uniq_imgs = {}
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
            Implemented by each child dataset.
        Returns:
            A transform function to be applied in the PIL images.
        """
        return None
