import os

import torch

from .video_stream import VideoStream


class KineticsStream(VideoStream):

    def _get_test_target(self, record, offsets):
        """
        Args:
            record : VideoRecord object
            offsets : List of image indices to be loaded from a video.
        Returns:
            target: Dict with the binary list of labels from a video and its relative path.
        """
        num_clips = int(len(offsets) / self.sample_frames)
        target = torch.full((num_clips,), int(record.label), dtype=torch.int32)

        video_name = os.path.join(
             os.path.basename(os.path.dirname(record.path)),
             os.path.basename(record.path))

        clip_paths = []
        for i_clip in range(num_clips):
            start = self.sample_frames * i_clip
            clip_offsets = offsets[start: start + self.sample_frames]

            last_frame = clip_offsets[-1]
            clip_paths.append('{}_{:06d}'.format(video_name, last_frame))

        return {'target': target, 'video_path': clip_paths}
