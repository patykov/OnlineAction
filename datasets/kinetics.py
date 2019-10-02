from .video_dataset import VideoDataset


class Kinetics(VideoDataset):
    """ Kinetics-400 Dataset """
    num_classes = 400
    multi_label = False

    def _parse_list(self):
        """
        Parses the annotation file to create a list of the videos relative path and their labels
        in the format: [label, video_path].
        """
        video_list = [x.strip().split(' ') for x in open(self.list_file)]

        if self.subset:  # Subset for tests!!!
            video_list = [v for i, v in enumerate(video_list) if i % 500 == 0]

        self.video_list = video_list

    def _get_train_target(self, record, *args):
        """
        Args:
            record : VideoRecord object
        Returns:
            target: Dict with the video label.
        """
        return {'target': int(record.label)}

    def _get_test_target(self, record):
        return self._get_train_target(record)
