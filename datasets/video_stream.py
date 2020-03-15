import numpy as np

from datasets.video_dataset import VideoDataset
from datasets.video_record import VideoRecord


class VideoStream(VideoDataset):

    def __init__(self, video_path, label, sample_frames, num_classes=157, transform=None,
                 mode='stream_centerCrop', clip_length=3):
        self.record = VideoRecord(video_path, label)
        self.sample_frames = sample_frames
        self.clip_length = clip_length  # in seconds
        self.num_classes = num_classes
        self.mode = mode

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.default_transforms()

        indices = self._get_test_indices(self.record)

        self.target = self._get_test_target(self.record, indices)
        self.total = self.target['target'].shape[0]

        # It's done here to better explore single image loading
        imgs_per_batch = 10
        self.internal_batch_size = (
            imgs_per_batch if 'centerCrop' in self.mode else int(imgs_per_batch/3))

        self.chunk_target = divide_chunks(self.target['target'], self.internal_batch_size)
        self.chunk_labels = divide_chunks(self.target['video_path'], self.internal_batch_size)
        self.chunk_indices = divide_chunks(divide_chunks(indices, self.sample_frames),
                                           self.internal_batch_size)

    def _get_test_indices(self, record):
        """
        Args:
            record : VideoRecord object
        Returns:
            offsets : List of image indices to be loaded from a video.
        """
        expanded_sample_length = int(self.clip_length * record.fps)
        sample_end_pos = list(range(expanded_sample_length, record.num_frames))
        offsets = []
        for p in sample_end_pos:
            offsets.extend(
                np.linspace(max(p - expanded_sample_length, 0),
                            min(p, record.num_frames - 1),
                            self.sample_frames,
                            endpoint=True, dtype=int))

        return np.array(offsets)

    def _get_test_target(self, record, offsets):
        """
        Args:
            record : VideoRecord object
            offsets : List of image indices to be loaded from a video.
        Returns:
            target: Dict with the binary list of labels from a video and its relative path.
        """
        raise NotImplementedError()

    def __getitem__(self, index):
        chunk_ids = self.chunk_indices[index]
        ids = [item for sublist in chunk_ids for item in sublist]
        data = self.get(self.record, ids)
        return data, {'target': self.chunk_target[index], 'video_path': self.chunk_labels[index]}

    def __len__(self):
        return len(self.chunk_target)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Video Path: {}\n'.format(self.record.path)
        fmt_str += '    Number of frames: {}\n'.format(self.record.num_frames)
        fmt_str += '    Video fps: {}\n'.format(self.record.fps)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))

        return fmt_str


def divide_chunks(my_list, n):
    return [my_list[i * n:(i + 1) * n] for i in range((len(my_list) + n - 1) // n)]
