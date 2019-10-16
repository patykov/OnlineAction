import os

import numpy as np
import torch

from datasets.video_dataset import VideoDataset
from datasets.video_record import VideoRecord


class VideoStream(VideoDataset):

    def __init__(self, video_path, label, num_classes=157, sample_frames=32, transform=None,
                 mode='test'):
        self.record = VideoRecord(video_path, label)
        self.sample_frames = sample_frames
        self.clip_length = 3  # in seconds
        self.num_classes = num_classes
        self.mode = mode

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()

        indices, first_frame = self._get_test_indices(self.record)
        self.target = self._get_test_target(self.record, indices)
        self.first_frame = first_frame
        self.total = self.target['target'].shape[0]

        # It's done here to better exploit single image loading
        self.internal_batch_size = 20 if self.mode == 'test' else 60

        self.chunk_target = divide_chunks(self.target['target'], self.internal_batch_size)
        self.chunk_indices = divide_chunks(
            np.transpose(np.split(np.array(indices), self.sample_frames)),
            self.internal_batch_size)

    def _get_test_indices(self, record):
        """
        Args:
            record : VideoRecord object
        Returns:
            offsets : List of image indices to be loaded from a video.
        """
        expanded_sample_length = int(self.clip_length * record.fps)
        sample_start_pos = list(range(expanded_sample_length, record.num_frames))
        offsets = []
        for p in sample_start_pos:
            offsets.extend(
                np.linspace(max(p - expanded_sample_length, 0),
                            min(p, record.num_frames - 1),
                            self.sample_frames,
                            dtype=int))

        return offsets, expanded_sample_length

    def _get_test_target(self, record, offsets):
        """
        Args:
            record : VideoRecord object
            offsets : List of image indices to be loaded from a video.
        Returns:
            target: Dict with the binary list of labels from a video and its relative path.
        """
        num_clips = int(len(offsets) / self.sample_frames)
        target = torch.IntTensor(num_clips, self.num_classes).zero_()
        for i_clip in range(num_clips):
            last_frame = offsets[self.sample_frames * i_clip + self.sample_frames - 1]
            for l in record.label:
                if l['start'] < last_frame / float(record.fps) < l['end']:
                    target[i_clip, int(l['class'][1:])] = 1

        return {'target': target, 'video_path': os.path.splitext(os.path.basename(record.path))[0]}

    def __getitem__(self, index):
        chunk_ids = self.chunk_indices[index]
        ids = [item for sublist in chunk_ids for item in sublist]
        data = self.get(self.record, ids)
        return data, self.chunk_target[index]

    def __len__(self):
        return len(self.chunk_target)


def divide_chunks(my_list, n):
    return [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]
