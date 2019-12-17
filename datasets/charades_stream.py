import os

import torch

from .video_stream import VideoStream


class CharadesStream(VideoStream):
    label_per_clip = False

    def _get_test_target(self, record, offsets):
        """
        Args:
            record: VideoRecord object
            offsets: List of image indices to be loaded from a video.
        Returns:
            target: Dict with the binary list of labels from a video and its relative path.
        """
        num_clips = int(len(offsets) / self.sample_frames)
        target = torch.zeros((num_clips, self.num_classes), dtype=torch.int8)
        video_name = os.path.splitext(os.path.basename(record.path))[0]
        clip_paths = []
        for i_clip in range(num_clips):
            start = self.sample_frames * i_clip
            clip_offsets = offsets[start: start + self.sample_frames]

            last_frame = clip_offsets[-1]
            clip_paths.append('{}_{:06d}'.format(video_name, last_frame))

            if self.label_per_clip:
                for frame in clip_offsets:
                    for l in record.label:
                        if l['start'] < frame / float(record.fps) < l['end']:
                            target[i_clip, int(l['class'][1:])] = 1

            else:
                for l in record.label:
                    if l['start'] < last_frame / float(record.fps) < l['end']:
                        target[i_clip, int(l['class'][1:])] = 1

        return {'target': target, 'video_path': clip_paths}
