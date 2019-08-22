import cv2
from PIL import Image


class VideoRecord(object):
    def __init__(self, video_path, label):
        self.path = video_path
        self.video = cv2.VideoCapture(self.path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.num_frames = self._get_num_frames()
        self.label = label

    def _get_num_frames(self):
        count = 0
        success, frame = self.video.read()
        if not success:
            print('Failed to load video {}'.format(self.path))
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
