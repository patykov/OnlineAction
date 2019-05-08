import os

import cv2
import jpeg4py as jpeg
import numpy as np
import torch.utils.data as data
from numpy.random import randint
from PIL import Image

import transforms as t


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
            indices : sorted list of frames indices
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
    def __init__(self, root_path, list_file, sample_frames=32,
                 image_tmpl='frame_{:06d}.jpg', transform=None,
                 train_mode=True, test_clips=10):
        self.root_path = root_path
        self.list_file = list_file
        self.sample_frames = sample_frames
        self.image_tmpl = image_tmpl
        self.train_mode = train_mode
        if not self.train_mode:
            self.num_clips = test_clips

        if transform is not None:
            self.transform = transform
        else:
            self.transform = t.GroupToTensorStack()

        self._parse_list()

    def _parse_list(self):
        self.video_list = [x.strip().split(' ') for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        expanded_sample_length = self.sample_frames * 4  # in order to drop every other frame
        if record.num_frames >= expanded_sample_length:
            start_pos = randint(record.num_frames - expanded_sample_length + 1)
            offsets = range(start_pos, start_pos + expanded_sample_length, 4)
        elif record.num_frames > self.sample_frames*2:
            start_pos = randint(record.num_frames - self.sample_frames*2 + 1)
            offsets = range(start_pos, start_pos + self.sample_frames*2, 2)
        elif record.num_frames > self.sample_frames:
            start_pos = randint(record.num_frames - self.sample_frames + 1)
            offsets = range(start_pos, start_pos + self.sample_frames, 1)
        else:
            offsets = np.sort(randint(record.num_frames, size=self.sample_frames))

        offsets = [int(v) for v in offsets]
        return offsets

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.sample_frames*2 + 1) / float(self.num_clips)
        sample_start_pos = np.array([int(tick * x) for x in range(self.num_clips)])
        offsets = []
        for p in sample_start_pos:
            offsets.extend(range(p, p+self.sample_frames*2, 2))

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
        record = VideoRecord(os.path.join(self.root_path, video_path))

        if self.train_mode:
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


class VideoFrameRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def label(self):
        return int(self._data[0])

    @property
    def path(self):
        return self._data[1]

    @property
    def num_frames(self):
        return int(self._data[2])


class VideoFrameDataset(VideoDataset):
    def __init__(self, root_path, list_file, sample_frames=32,
                 image_tmpl='frame_{:06d}.jpg', transform=None,
                 train_mode=True, test_clips=10):
        VideoDataset.__init__(self, root_path, list_file, sample_frames, image_tmpl,
                              transform, train_mode, test_clips)

    def _load_image(self, video_dir, idx):
        img_path = os.path.join(self.root_path, video_dir, self.image_tmpl.format(idx))
        try:
            # Loading images with PIL is slower.
            return [Image.fromarray(jpeg.JPEG(img_path).decode()).convert('RGB')]
        except IOError:
            print("Couldn't load image:{}".format(img_path))
            return None

    def _parse_list(self):
        self.video_list = [VideoFrameRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.video_list[index]

        if self.train_mode:
            segment_indices = self._sample_indices(record)
            process_data, label = self.get(record, segment_indices)
            while process_data is None:
                index = randint(0, len(self.video_list) - 1)
                process_data, label = self.__getitem__(index)
        else:
            segment_indices = self._get_test_indices(record)
            process_data, label = self.get(record, segment_indices)
            if process_data is None:
                raise ValueError('sample indices:', record.path, segment_indices)

        data = process_data.squeeze(0)
        data = data.view(3, -1, self.sample_frames, data.size(2), data.size(3)).contiguous()
        data = data.permute(1, 0, 2, 3, 4).contiguous()

        return data, label

    def get(self, record, indices):
        uniq_imgs = {}
        uniq_id = np.unique(indices)
        for ind in uniq_id:
            seg_img = self._load_image(record.path, ind)
            if seg_img is None:
                return None, None
            uniq_imgs[ind] = seg_img

        images = [uniq_imgs[i][0] for i in indices]
        images = self.transform(images)
        return images, record.label
