import cv2
from PIL import Image


class VideoRecord(object):
    def __init__(self, video_path, label, reliable=False):
        self.path = video_path
        self.label = label
        self.video = cv2.VideoCapture(self.path)
        self.num_frames = self._get_num_frames(reliable)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)

    def _get_num_frames(self, reliable):
        success, frame = self.video.read()
        if not success:
            print('Failed to load video {}'.format(self.path))
            return 0

        if reliable:
            # Fastest and easiest way to get video frame count. However, it does not work for
            # Kinetics dataset, where most videos have a mismatch in the frame count.
            count = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            # Counting manually
            count = 0
            while(success):
                success, frame = self.video.read()
                count += 1

        self.video.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)  # Set video to the start
        return count

    def get_frames(self, indices):
        """
        Args:
            indices : Sorted list of frames indices
        Returns:
            images : Dictionary in the format: {frame_id: PIL Image}
        """
        images = dict()
        self.video.set(cv2.CAP_PROP_POS_FRAMES, min(indices))
        for count in range(min(indices), max(indices)+1):
            success, frame = self.video.read()
            if not success:
                print('Could not load frame {} from video {} (num_frames: {})\n'.format(
                    count, self.path, self.num_frames))
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if count in indices:
                images[count] = Image.fromarray(frame)

        return images
