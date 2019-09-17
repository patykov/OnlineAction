import cv2
from PIL import Image


class VideoRecord(object):
    def __init__(self, video_path, label, reliable=False):
        self.path = video_path
        self.label = label
        self.num_frames, self.fps = self._get_video_data(reliable)

    def _get_video(self):
        return cv2.VideoCapture(self.path)

    def _get_video_data(self, reliable):
        video = self._get_video()
        success, _ = video.read()
        if not success:
            raise ValueError('Failed to load video {}'.format(self.path))

        fps = video.get(cv2.CAP_PROP_FPS)
        if reliable:
            # Fastest and easiest way to get video frame count. However, it does not work for
            # Kinetics dataset, where most videos have a mismatch in the frame count.
            count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            # Counting manually
            count = 0
            while(success):
                success, _ = video.read()
                count += 1

        video.release()
        return count, fps

    def get_frames(self, indices):
        """
        Args:
            indices : Sorted list of frames indices
        Returns:
            images : Dictionary in the format: {frame_id: PIL Image}
        """
        images = dict()
        video = self._get_video()
        for count in range(max(indices) + 1):
            success, frame = video.read()
            if not success:
                raise ValueError('Could not load frame {} from video {} (num_frames: {})\n'.format(
                    count, self.path, self.num_frames))

            if count in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images[count] = Image.fromarray(frame)

        video.release()
        return images
