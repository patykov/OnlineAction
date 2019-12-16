import os

import numpy as np
import torch
from torch.nn import MaxPool1d

import metrics.charades_classify as cc
from datasets.get import get_dataloader, get_dataset


def divide_per_video(ids, targets):
    videos = {}

    video_name = None
    for i, video_frame in enumerate(ids):
        name = video_frame.split('_')[0]
        if name != video_name:
            # new video!
            videos[name] = {
                'ids': [],
                'targets': []
            }
            video_name = name

        videos[video_name]['ids'].append(video_frame)
        videos[video_name]['targets'].append(targets[i])

    return videos


def divide_per_clip(ids, classes):
    clips_ids = []
    clips_classes = []

    video_name = None
    for i, video_frame in enumerate(ids):
        name = video_frame.split('_')[0]
        if name != video_name:
            # new video! But first, save old video
            if i > 0:
                clips_ids.append(np.array(video_ids))
                clips_classes.append(np.array(video_classes))
            # star new one
            video_ids = []
            video_classes = []
            video_name = name

        video_ids.append(video_frame)
        video_classes.append(classes[i])

    # Append last video
    clips_ids.append(np.array(video_ids))
    clips_classes.append(np.array(video_classes))

    return np.array(clips_ids), np.array(clips_classes)


def video_output(outputs):
    num_clips, num_classes = outputs.shape
    max_pool = MaxPool1d(num_clips)

    outputs = torch.tensor(outputs)

    data = outputs.view(1, -1, num_classes).contiguous()
    data = data.permute(0, 2, 1).contiguous()

    data = max_pool(data)
    video_data = data.view(num_classes).contiguous()

    return video_data


def select_n_clips(video_classes, n=10):
    num_frames = len(video_classes)
    ids = np.linspace(0, num_frames-1, n, dtype=int)

    return video_classes[ids]


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

        video_stream = get_dataloader(
            (dataset, 'stream'), video_path=video_path, label=label, batch_size=1,
            num_classes=num_classes, mode=mode, distributed=False, num_workers=0)

        video_data = []
        video_target = []
        for chunk_data, chunk_target in video_stream:
            video_data.append(chunk_data[:, :, :, -1, :, :])
            video_target.append(chunk_target)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_fps = video_stream.dataset.record.fps

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

    # Dividing per clip
    _, gt_clips_classes = divide_per_clip(gt_ids, gt_classes)
    _, test_clips_classes = divide_per_clip(test_ids, test_scores)

    # Get classes thresholds
    test_classes_n = [select_n_clips(np.array(clip_data), n=10) for clip_data in test_clips_classes]

    test_clip_n_mean = np.array([video_output(t_c).numpy() for t_c in test_classes_n])
    gt_clips_n_mean = np.array([(sum(gt_c) > 0).astype(int) for gt_c in gt_clips_classes])

    thresholds = cc.get_thresholds(test_clip_n_mean, gt_clips_n_mean)

    return get_video, videos, thresholds, {
        'mean': video_dataset.input_mean,
        'std': video_dataset.input_std}
