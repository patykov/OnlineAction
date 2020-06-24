import argparse
import os
import re

import numpy as np
import torch
from scipy import signal
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


def adapt_predictions(gt_array, test_classes, classes_file):
    class_to_new = {}
    new_to_class = {}
    with open(classes_file, 'r') as file:
        for line in file:
            class_match = re.match(r'(\d*) (.*) (\d*)', line)
            if class_match:
                old_class_id, new_class_id, _ = class_match.groups()
            class_to_new[int(old_class_id)] = int(new_class_id)
            new_to_class.setdefault(int(new_class_id), []).append(int(old_class_id))

    num_classes = len(np.unique(list(class_to_new.values())))
    new_gt_array = np.zeros((gt_array.shape[0], num_classes))
    mean_test_array = np.zeros((gt_array.shape[0], num_classes))
    max_test_array = np.zeros((gt_array.shape[0], num_classes))
    print(new_gt_array.shape)
    for frame_id, frame_gt in enumerate(gt_array):
        old_classes = np.where(frame_gt)[0]
        new_classes = [class_to_new[c_id] for c_id in old_classes]
        new_gt_array[frame_id, new_classes] = 1

        mean_temp_array = []
        max_temp_array = []
        for new_c in range(num_classes):
            preds = test_classes[frame_id, new_to_class[new_c]]
            mean_temp_array.append(np.mean(preds))
            max_temp_array.append(np.max(preds))

        mean_test_array[frame_id, :] = mean_temp_array
        max_test_array[frame_id, :] = max_temp_array

    return new_gt_array, mean_test_array, max_test_array


def main(classes_file, gt_file, results_dir, files_dir, output_dir,
         verb_classes_file, object_classes_file):
    # Getting output files
    output_path = os.path.join(output_dir, files_dir)
    ids_file = output_path + '_ids.npy'
    preds_file = output_path + '_preds.npy'

    if os.path.exists(ids_file) and os.path.exists(preds_file):
        print('Loading saved test values!')
        print(files_dir)
        test_ids = np.load(ids_file)
        test_classes = np.load(preds_file)

    else:
        # Getting all results files
        result_files = sorted(
            [os.path.join(results_dir, f) for f in os.listdir(
                results_dir) if f.endswith('.txt')])

        # Reading all results
        test_ids = []
        test_classes = []
        for fname in result_files:
            print(fname)
            v_ids, v_scores = cc.read_causal_file(fname)
            test_ids += v_ids
            test_classes += v_scores
            print(len(test_ids), len(test_classes))

        # Saving results to numpy files
        np.save(ids_file, test_ids)
        np.save(preds_file, test_classes)

    # Reading GT file
    gt_ids, gt_classes = cc.read_file(gt_file)
    gt_classes = np.array(gt_classes)
    n_test = len(gt_ids)

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

    # per frame using mean of all clips
    mAP, _, _, _, _, cAP, _ = cc.charades_map(test_classes, gt_classes)
    print('Per frame:')
    print('mAP: {:4.1%}, cAP: {:4.1%}'.format(mAP, cAP))

    # per clip result
    test_classes_n = [select_n_clips(np.array(clip_data), n=10) for clip_data in test_clips_classes]

    test_clip_n_mean = np.array([video_output(t_c).numpy() for t_c in test_classes_n])
    gt_clips_n_mean = np.array([(sum(gt_c) > 0).astype(int) for gt_c in gt_clips_classes])

    mAP_10, _, _, _, _, cAP_10, _ = cc.charades_map(test_clip_n_mean, gt_clips_n_mean)
    print('Per clip:')
    print('mAP: {:4.1%}, cAP: {:4.1%}\n'.format(mAP_10, cAP_10))

    print('\n\nPer-frame results:')
    # Verbs
    gt_verb, mean_test_verb, max_test_verb = adapt_predictions(
        gt_classes, test_classes, verb_classes_file)
    mAP_n_v_mean, _, _, _, _, cAP_n_v_mean, _ = cc.charades_map(mean_test_verb, gt_verb)
    print('Mean verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_mean, cAP_n_v_mean))
    mAP_n_v_max, _, _, _, _, cAP_n_v_max, _ = cc.charades_map(max_test_verb, gt_verb)
    print('Max verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_max, cAP_n_v_max))

    # Objects
    gt_obj, mean_test_obj, max_test_obj = adapt_predictions(
        gt_classes, test_classes, object_classes_file)
    mAP_n_o_mean, _, _, _, _, cAP_n_o_mean, _ = cc.charades_map(mean_test_obj, gt_obj)
    print('Mean obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_mean, cAP_n_o_mean))
    mAP_n_o_max, _, _, _, _, cAP_n_o_max, _ = cc.charades_map(max_test_obj, gt_obj)
    print('Max obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_max, cAP_n_o_max))

    print('\nUsing enhanced per-frame result with mean')
    # per clip result using the N clip with mean of all frames
    test_targets_perF_mean, gt_targets_perF_mean = get_perF_result(
        gt_clips_ids, gt_clips_classes, test_clips_ids, test_clips_classes,
        'mean', delay_in_secs=3.0)

    # Verbs
    gt_verb, mean_test_verb, max_test_verb = adapt_predictions(
        gt_targets_perF_mean, test_targets_perF_mean, verb_classes_file)
    mAP_n_v_mean, _, _, _, _, cAP_n_v_mean, _ = cc.charades_map(mean_test_verb, gt_verb)
    print('Mean verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_mean, cAP_n_v_mean))
    mAP_n_v_max, _, _, _, _, cAP_n_v_max, _ = cc.charades_map(max_test_verb, gt_verb)
    print('Max verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_max, cAP_n_v_max))

    # Objects
    gt_obj, mean_test_obj, max_test_obj = adapt_predictions(
        gt_targets_perF_mean, test_targets_perF_mean, object_classes_file)
    mAP_n_o_mean, _, _, _, _, cAP_n_o_mean, _ = cc.charades_map(mean_test_obj, gt_obj)
    print('Mean obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_mean, cAP_n_o_mean))
    mAP_n_o_max, _, _, _, _, cAP_n_o_max, _ = cc.charades_map(max_test_obj, gt_obj)
    print('Max obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_max, cAP_n_o_max))

    ##########################################################################
    print('\n\nPer-clip results')
    gt_verb, mean_test_verb, max_test_verb = adapt_predictions(
        gt_clips_n_mean, test_clip_n_mean, verb_classes_file)
    mAP_n_v_mean, _, _, _, _, cAP_n_v_mean, _ = cc.charades_map(mean_test_verb, gt_verb)
    print('Mean verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_mean, cAP_n_v_mean))
    mAP_n_v_max, _, _, _, _, cAP_n_v_max, _ = cc.charades_map(max_test_verb, gt_verb)
    print('Max verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_max, cAP_n_v_max))

    # Objects
    gt_obj, mean_test_obj, max_test_obj = adapt_predictions(
        gt_clips_n_mean, test_clip_n_mean, object_classes_file)
    mAP_n_o_mean, _, _, _, _, cAP_n_o_mean, _ = cc.charades_map(mean_test_obj, gt_obj)
    print('Mean obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_mean, cAP_n_o_mean))
    mAP_n_o_max, _, _, _, _, cAP_n_o_max, _ = cc.charades_map(max_test_obj, gt_obj)
    print('Max obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_max, cAP_n_o_max))

    print('\nUsing enhanced per-frame result with mean')
    # per clip result using the N clip with mean of all frames
    test_targets_perF_mean, gt_targets_perF_mean = get_perF_result(
        gt_clips_ids, gt_clips_classes, test_clips_ids, test_clips_classes,
        'mean', delay_in_secs=3.0, frame=False)

    test_classes_n = [select_n_clips(
        np.array(clip_data), n=10) for clip_data in test_targets_perF_mean]

    test_clip_n_f = np.array([video_output(t_c).numpy() for t_c in test_classes_n])
    gt_clips_n_f = np.array([(sum(gt_c) > 0).astype(int) for gt_c in gt_targets_perF_mean])

    mAP_n_f, _, _, _, _, cAP_n_f, _ = cc.charades_map(test_clip_n_f, gt_clips_n_f)
    print('Normal | {:4.02%} | {:4.02%}'.format(mAP_n_f, cAP_n_f))

    # Verbs
    gt_verb, mean_test_verb, max_test_verb = adapt_predictions(
        gt_clips_n_f, test_clip_n_f, verb_classes_file)
    mAP_n_v_mean, _, _, _, _, cAP_n_v_mean, _ = cc.charades_map(mean_test_verb, gt_verb)
    print('Mean verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_mean, cAP_n_v_mean))
    mAP_n_v_max, _, _, _, _, cAP_n_v_max, _ = cc.charades_map(max_test_verb, gt_verb)
    print('Max verb | {:4.02%} | {:4.02%}'.format(mAP_n_v_max, cAP_n_v_max))

    # Objects
    gt_obj, mean_test_obj, max_test_obj = adapt_predictions(
        gt_clips_n_f, test_clip_n_f, object_classes_file)
    mAP_n_o_mean, _, _, _, _, cAP_n_o_mean, _ = cc.charades_map(mean_test_obj, gt_obj)
    print('Mean obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_mean, cAP_n_o_mean))
    mAP_n_o_max, _, _, _, _, cAP_n_o_max, _ = cc.charades_map(max_test_obj, gt_obj)
    print('Max obj | {:4.02%} | {:4.02%}'.format(mAP_n_o_max, cAP_n_o_max))


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
    parser.add_argument('--object_classes_file',
                        type=str,
                        default=None,
                        help='Full path to the file with the classes to verbs mapping')

    args = parser.parse_args()

    main(args.classes_file, args.gt_file, args.results_dir, args.files_dir, args.output_dir,
         args.verb_classes_file, args.object_classes_file)
