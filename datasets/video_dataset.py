import os

import cv2
import numpy as np
import torch.utils.data as data
from numpy.random import randint
from PIL import Image


class VideoRecord(object):
    def __init__(self, video_path):
        self.path = video_path
        self.video = cv2.VideoCapture(self.path)
        self.num_frames = self.get_num_frames()

    def get_num_frames(self):
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
            video_list : Object that can be indexed (video_list[id]) returning a tuple
                         (label, video_path) and returns the total number of videos when
                         len(video_list) is called. Just like a list of tuples.
        """
        return None

    def _sample_indices(self, record):
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
            Implemented by each child dataset.
        Argument:
            record : VideoRecord object
        Returns:
            offsets : List of image indices to be loaded
        """
        return None

    def __getitem__(self, index):
        label, video_path = self.video_list[index]
        record = VideoRecord(os.path.join(self.root_path, video_path))

        if self.mode == 'train':
            segment_indices = self._sample_indices(record)
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

        return data, int(label)

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
