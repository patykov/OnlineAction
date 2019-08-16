import os

import pytest
import torch

from datasets.charades import Charades
from datasets.kinetics import Kinetics


@pytest.mark.parametrize('mode', ['train', 'val', 'test'])
def test_charades_data_list(mode):
    abs_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(abs_dir, 'data', 'charades')

    dataset = Charades(
        os.path.join(data_dir, 'videos'),
        os.path.join(data_dir, 'list_file.csv'),
        mode=mode)

    assert len(dataset) == 2

    correct_paths = ['19UA5', 'S6MPZ']
    correct_actions = [
        [31, 20, 106, 26],
        [9, 11, 15, 19, 156, 59, 61, 61, 17, 63]]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (data, label) in enumerate(data_loader):

        assert label['video_path'][0] == correct_paths[i]

        correct_target = torch.IntTensor(dataset.num_classes).zero_()
        correct_target[correct_actions[i]] = 1

        if mode == 'train':
            # Randomly selects a clip from the video, thus, might not cover all actions
            assert sum(label['target'][0]) <= sum(correct_target)
        else:
            assert torch.equal(label['target'][0], correct_target)


@pytest.mark.parametrize('sample_frames', [8, 32])
def test_charades_get_frames(sample_frames):
    abs_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(abs_dir, 'data', 'charades')

    dataset = Charades(
        os.path.join(data_dir, 'videos'),
        os.path.join(data_dir, 'list_file.csv'),
        mode='val')

    assert dataset.sample_frames * dataset.stride == 64  # temporal input extesion
