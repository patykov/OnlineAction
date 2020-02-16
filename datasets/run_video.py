import os
from collections import defaultdict

import numpy as np

import metrics.charades_classify as cc
import torch
from datasets.get import get_dataloader, get_dataset


def divide_per_video(ids, targets):
    videos = defaultdict(lambda: defaultdict(list))

    for i, video_frame in enumerate(ids):
        video_name = video_frame.split('_')[0]

        videos[video_name]['ids'].append(video_frame)
        videos[video_name]['targets'].append(targets[i])

    return videos


def main(dataset, map_file, root_data_path, subset, sample_frames, gt_path, results_file,
         classes_file, mode):
    # Load data
    video_dataset = get_dataset(dataset, list_file=map_file, root_path=root_data_path,
                                subset=subset, mode='stream', sample_frames=sample_frames)
    num_classes = video_dataset.num_classes

    def get_video(vid=None):
        if vid is None:
            vid = np.random.randint(0, num_classes)
        video_path, label = video_dataset[vid]

        stream_dataset = get_dataset(
            (dataset, 'stream'), video_path=video_path, label=label,
            num_classes=num_classes, mode=mode)
        video_stream = get_dataloader(
            stream_dataset, batch_size=1, distributed=False, num_workers=0)

        video_data = []
        video_target = []
        for chunk_data, chunk_target in video_stream:
            video_data.append(chunk_data[:, :, :, -1, :, :])
            video_target.append(chunk_target)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_fps = stream_dataset.record.fps

        all_data = torch.from_numpy(np.hstack(video_data).squeeze(0))
        all_target = np.hstack([t['target'] for t in video_target]).squeeze(0)
        all_paths = np.vstack([t['video_path'] for t in video_target])

        return all_data, all_target, all_paths, video_name, video_fps, vid

    gt_ids, gt_classes = cc.read_file(gt_path)
    test_ids, test_scores = cc.read_file(results_file)

    gt_classes = np.array(gt_classes)
    test_scores = np.array(test_scores)

    n = len(gt_ids)
    n_test = len(test_ids)
    if n < n_test:
        # Check if its duplicate items
        test_ids, test_index_order = np.unique(test_ids, return_index=True)
        test_scores = test_scores[test_index_order]

    assert gt_classes.shape == test_scores.shape

    # Get videos clips
    videos = divide_per_video(test_ids, test_scores)

    # Get classes thresholds
    thresholds = cc.get_thresholds(test_scores, gt_classes)

    return get_video, videos, thresholds, {
        'mean': video_dataset.input_mean,
        'std': video_dataset.input_std}
