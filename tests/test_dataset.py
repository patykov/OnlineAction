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

        if mode == 'test':
            assert torch.equal(label['target'][0], correct_target)
        else:
            # Randomly selects a clip from the video, thus, might not cover all actions
            assert sum(label['target'][0]) <= sum(correct_target)


@pytest.mark.parametrize('mode', ['train', 'val', 'test'])
@pytest.mark.parametrize('sample_frames', [8, 32])
def test_charades_get_frames(sample_frames, mode):
    abs_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(abs_dir, 'data', 'charades')

    dataset = Charades(
        os.path.join(data_dir, 'videos'),
        os.path.join(data_dir, 'list_file.csv'),
        sample_frames=sample_frames,
        mode=mode)

    assert dataset.sample_frames * dataset.stride == 64  # temporal input extesion

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    for i, (data, label) in enumerate(data_loader):
        target = label['target']
        if mode == 'test':
            assert data.shape == (2, dataset.test_clips*3, 3, sample_frames, 224, 224)
            assert target.shape == (2, dataset.num_classes)

        else:
            assert data.shape == (2, 1, 3, sample_frames, 224, 224)
            assert target.shape == (2, dataset.num_classes)


@pytest.mark.parametrize('mode', ['train', 'val', 'test'])
def test_kinetics_data_list(mode):
    abs_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(abs_dir, 'data', 'kinetics')

    dataset = Kinetics(
        os.path.join(data_dir, 'videos'),
        os.path.join(data_dir, 'list_file.txt'),
        mode=mode)

    assert len(dataset) == 2

    correct_actions = [0, 1]
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (data, label) in enumerate(data_loader):
        assert label['target'][0] == correct_actions[i]


@pytest.mark.parametrize('mode', ['train', 'val', 'test'])
@pytest.mark.parametrize('sample_frames', [8, 32])
def test_kinetics_get_frames(sample_frames, mode):
    abs_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(abs_dir, 'data', 'kinetics')

    dataset = Kinetics(
        os.path.join(data_dir, 'videos'),
        os.path.join(data_dir, 'list_file.txt'),
        sample_frames=sample_frames,
        mode=mode)

    assert dataset.sample_frames * dataset.stride == 64  # temporal input extesion

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    for i, (data, label) in enumerate(data_loader):
        target = label['target']
        if mode == 'test':
            assert data.shape == (2, dataset.test_clips*3, 3, sample_frames, 256, 256)
            assert list(target.shape) == [2]

        else:
            assert data.shape == (2, 1, 3, sample_frames, 224, 224)
            assert list(target.shape) == [2]
