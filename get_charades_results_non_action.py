import argparse
import os
import re
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import MaxPool1d

import metrics.charades_classify as cc


def divide_per_clip(ids, classes):
    clips_ids = []
    clips_classes = []

    video_ids = []
    video_classes = []
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


def get_perF_result(gt_clips_ids, gt_clips_classes, test_clips_ids, test_clips_classes,
                    method='mean', delay_in_secs=3.0, secs_in_clip=3, frame=True):
    test_clips_classes_perF = []
    gt_clips_classes_perF = []

    for v in range(gt_clips_ids.shape[0]):
        v_gt_ids = gt_clips_ids[v]
        v_gt_targets = gt_clips_classes[v]
        v_test_ids = test_clips_ids[v]
        v_test_targets = test_clips_classes[v]

        assert np.all(v_gt_ids == v_test_ids), (v_gt_ids, v_test_ids)

        new_test_targets = []
        new_gt_targets = []

        clip_length = int(re.match(r'(.*)_(\d*)', v_test_ids[0]).groups()[-1]) + 1  # in # of frames
        fps = float(clip_length/secs_in_clip)
        num_clips = int(fps * delay_in_secs) + 1
        num_frames = len(v_gt_ids)
        test_frames = len(v_test_targets)

        if num_clips > test_frames:
            continue

        if method == 'gaussian':
            norm_weights = signal.gaussian(clip_length, std=clip_length/4)
            middle = int(clip_length / 2.0)

        for f_id in range(num_frames):
            new_gt_targets.append(v_gt_targets[f_id])
            clips_targets = v_test_targets[f_id:(f_id + num_clips)]

            if method == 'gaussian':
                norm = norm_weights[:clips_targets.shape[0]]

                if clips_targets.shape[0] > len(norm):
                    clips_targets = clips_targets[:len(norm), :]

                clips_targets = np.swapaxes(clips_targets, 0, 1)
                try:
                    new_test_targets.append(np.average(clips_targets, axis=1, weights=norm))
                except ValueError:
                    print(norm_weights.shape, clip_length, middle)
                    print(clips_targets.shape, norm.shape)

            elif method == 'mean':
                new_test_targets.append(np.mean(clips_targets, axis=0))

            elif method == 'median':
                new_test_targets.append(np.median(clips_targets, axis=0))

            elif method == 'max_pool':
                outputs = torch.tensor(clips_targets)
                data, _ = outputs.max(0)
                new_test_targets.append(data.numpy())

        test_clips_classes_perF.append(new_test_targets)
        gt_clips_classes_perF.append(new_gt_targets)

    if frame:
        test_clips_classes_perF = np.vstack(test_clips_classes_perF)
        gt_clips_classes_perF = np.vstack(gt_clips_classes_perF)

    return test_clips_classes_perF, gt_clips_classes_perF


def get_per_delay_result(gt_clips_ids, gt_clips_classes, test_clips_ids, test_clips_classes,
                         method='mean', delay_in_secs=3.0, secs_in_clip=3, frame=True,
                         enhanced=False):
    test_clips_classes_per_delay = []
    gt_clips_classes_per_delay = []
    total_length = 0

    for v in range(gt_clips_ids.shape[0]):
        v_gt_ids = gt_clips_ids[v]
        v_gt_targets = gt_clips_classes[v]
        v_test_ids = test_clips_ids[v]
        v_test_targets = test_clips_classes[v]

        assert np.all(v_gt_ids == v_test_ids), (v_gt_ids, v_test_ids)

        new_test_targets = []
        new_gt_targets = []

        clip_length = int(re.match(r'(.*)_(\d*)', v_test_ids[0]).groups()[-1]) + 1  # in # of frames
        last_frame = int(re.match(r'(.*)_(\d*)', v_test_ids[-1]).groups()[-1]) + 1  # in # of frames
        fps = float(clip_length/secs_in_clip)
        num_clips = int(fps * delay_in_secs) + 1
        num_frames = len(v_gt_ids)
        test_frames = len(v_test_targets)

        total_length += (last_frame - clip_length) / fps

        if num_clips > test_frames:
            continue

        for f_id in range(0, num_frames, num_clips):
            clips_gt = v_gt_targets[f_id:(f_id + num_clips)]
            clips_targets = v_test_targets[f_id:(f_id + num_clips)]

            num_f_in_delay = len(clips_targets)
            frames_ids = np.linspace(0, num_f_in_delay-1, min(10, num_f_in_delay), dtype=int)
            temp_clips_targets = clips_targets
            if enhanced:
                # enhancing over enhanced....
                for i, fid in enumerate(frames_ids):
                    group_targets = clips_targets[fid:fid + num_clips]
                    if method == 'mean':
                        temp_clips_targets[fid] = np.mean(group_targets, axis=0)

                    elif method == 'max_pool':
                        outputs = torch.tensor(group_targets)
                        data, _ = outputs.max(0)
                        temp_clips_targets[fid] = data.numpy()

            clips_targets = temp_clips_targets[frames_ids]

            new_gt = (clips_gt.sum(axis=0) > 1).astype(int)
            new_gt_targets.append(new_gt)

            if method == 'mean':
                new_test_targets.append(np.mean(clips_targets, axis=0))

            elif method == 'max_pool':
                outputs = torch.tensor(clips_targets)
                data, _ = outputs.max(0)
                new_test_targets.append(data.numpy())

        test_clips_classes_per_delay.append(new_test_targets)
        gt_clips_classes_per_delay.append(new_gt_targets)

    if frame:
        test_clips_classes_per_delay = np.vstack(test_clips_classes_per_delay)
        gt_clips_classes_per_delay = np.vstack(gt_clips_classes_per_delay)

    return test_clips_classes_per_delay, gt_clips_classes_per_delay, total_length



def main(classes_file, gt_file, results_dir, files_dir, output_dir, verb_classes_file):
    # Getting output files
    output_path = os.path.join(output_dir, files_dir)
    ids_file = output_path + '_ids.npy'
    preds_file = output_path + '_preds.npy'

    print('Loading saved test values!')
    test_ids = np.load(ids_file)
    test_classes = np.load(preds_file)

    # Reading GT file
    gt_ids, gt_classes = cc.read_file(gt_file)
    gt_classes = np.array(gt_classes)
    n_test = len(gt_ids)
    if verb_classes_file:
        gt_classes = cc.get_gt_per_verb(gt_classes, verb_classes_file)

    # Check if there are duplicate items and SORT items
    test_ids, test_index_order = np.unique(test_ids, return_index=True)
    test_classes = np.array(test_classes)[test_index_order]

    # Asserting same lenght in gt and test
    if n_test != len(test_ids):
        print('Wrong number of samples! Expected {} frames, found {}'.format(
            n_test, len(test_ids)))

        # Exclude missing frames from gt
        equal = set(test_ids)
        missing_ids = [i for i, x in enumerate(gt_ids) if x not in equal]
        for m_id in sorted(missing_ids, reverse=True):
            del gt_ids[m_id]
        gt_classes = np.delete(gt_classes, missing_ids, axis=0)
        print('Removed {} frames from GT'.format(len(missing_ids)))
        print('GT ids: {}, GT classes: {}, test ids: {}, test classes: {}'.format(
            len(gt_ids), gt_classes.shape, len(test_ids), test_classes.shape
        ))

    assert np.all(gt_ids == test_ids)

    # Dividing per clip
    gt_clips_ids, gt_clips_classes = divide_per_clip(gt_ids, gt_classes)
    test_clips_ids, test_clips_classes = divide_per_clip(test_ids, test_classes)

    # Loading thresholds
    thr_m_ap = np.load(output_path + '_thr_m_ap.npy')
    thr_m_pr = np.load(output_path + '_thr_m_pr.npy')

    new_thr = np.array([tc.mean(axis=0) for tc in test_clips_classes]).mean(axis=0)
    print(new_thr.shape)
    new_thr = new_thr * 0.9

    print('Non-action results')
    for thr, thr_name in zip([thr_m_ap, thr_m_pr, new_thr], ['thr_m_ap', 'thr_m_pr', 'new_thr']):
        # Applying threshold
        new_test_clips_classes = np.array([(cc > thr).astype(int) for cc in test_clips_classes])

        delays = [1.0, 3.0, 5.0]
        print('{} | {:6} | {:6} | {:6} | {:4}'.format(
            'delay', 'Acc 0', 'Acc 1', 'Mean acc', 'time per frame (ms)'))

        for d in delays:
            begin_time = time.time()
            test_targets_per_delay, gt_targets_per_delay, _ = get_per_delay_result(
                gt_clips_ids, gt_clips_classes, test_clips_ids, new_test_clips_classes,
                'mean', delay_in_secs=d)
            end_time = time.time()

            gt_vector = []
            pred_vector = []
            n_periods = gt_targets_per_delay.shape[0]
            for p_id in range(n_periods):
                gt = (sum(gt_targets_per_delay[p_id, :]) > 0).astype(int)
                pred = (sum(test_targets_per_delay[p_id, :]) > 0).astype(int)

                gt_vector.append(gt)
                pred_vector.append(pred)

            gt_vector = np.array(gt_vector)
            pred_vector = np.array(pred_vector)
            correct = sum(gt_vector == pred_vector)
            acc = correct / len(gt_vector)

            cf = confusion_matrix(gt_vector, pred_vector).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            acc = cls_hit / cls_cnt
            m_acc = np.nanmean(acc)

            print('{:4}s | {:4.1%} | {:4.1%} | {:4.1%} | {:.4}'.format(
                d, acc[0], acc[1], m_acc, (end_time - begin_time)*1000/len(gt_ids)))

        # tpr_file = output_path + '_tpr_{}_results'.format(thr_name)
        # np.save(tpr_file, tpr_d)


if __name__ == '__main__':
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes_file', type=str)
    parser.add_argument('--gt_file', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--files_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--verb_classes_file',
                        type=str,
                        default=None,
                        help='Full path to the file with the classes to verbs mapping')

    args = parser.parse_args()

    main(args.classes_file, args.gt_file, args.results_dir, args.files_dir, args.output_dir,
         args.verb_classes_file)
