import csv
import os
import time

import numpy as np

from datasets.video_stream import VideoStream


def charades_v1_classify(cls_file, gt_path, per_frame=False, calibrated=False):
    """
        Evaluate charades dataset for multi-label classification per video.
        Adapted from Charades_v1_classify.m code from charades dataset providers.

    Argument:
            cls_file: path of the input file with the classification scores
            gt_path: the path of the groundtruth file
        Returns:
            rec_all: recall
            prec_all: precision
            ap_all: AP for each class
            map: MAP

    """
    t1 = time.time()
    gt_ids, gt_classes = read_file(gt_path) if per_frame else load_charades(gt_path)
    w_array = get_w_array(gt_classes) if calibrated else None
    n_test = len(gt_ids)

    t2 = time.time()
    print('Load GT file in {}'.format(t2-t1))

    # Load test scores
    test_ids, test_scores = read_file(cls_file)

    t3 = time.time()
    print('Load test scores file in {}'.format(t3-t2))

    n = len(test_scores)
    if n < n_test:
        print('Warning: {} items missing\n'.format(n_test-n))
        # For partial evaluation
        subset_gt_classes = []
        for gt_i, gt_id in enumerate(gt_ids):
            if gt_id in test_ids:
                subset_gt_classes.append(gt_classes[gt_i])
        gt_classes = subset_gt_classes
    elif n_test < n:
        # Check if its duplicate items
        test_ids, test_index_order = np.unique(test_ids, return_index=True)
        test_scores = np.array(test_scores)[test_index_order]
        n = len(test_scores)
        if n_test < n:
            raise RuntimeError('There are {} extra items!'.format(n-n_test))

    t4 = time.time()
    print('Adjust files in {}'.format(t4-t3))

    mAP, wAP, ap = charades_map(np.array(test_scores), np.array(gt_classes), w_array)
    print('MAP: {:.02%}\n'.format(mAP))

    t5 = time.time()
    print('mAP in {}'.format(t5-t4))

    return mAP, wAP, ap


def map_func(submission_array, gt_array, w_array):
    """ Returns mAP, weighted mAP, and AP array """
    m_aps = []
    n_classes = submission_array.shape[1]
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        tp = gt_array[:, oc_i][sorted_idxs] == 1
        fp = np.invert(tp)
        n_pos = tp.sum()

        t_pcs = np.cumsum(tp)
        f_pcs = np.cumsum(fp)

        w_t_pcs = t_pcs * w_array[oc_i]
        prec = w_t_pcs / (f_pcs + w_t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.nanmean(m_aps)
    w_ap = np.nansum(m_aps * gt_array.sum(axis=0) / gt_array.sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array, w_array=None):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """

    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF
    if w_array is None:
        w_array = np.ones(submission_array.shape[1])

    return map_func(fix, gt_array, w_array)


def read_file(file_path):
    with open(file_path, 'r') as file:
        text = file.readlines()

    split_text = [t.replace('\n', '').split(' ') for t in text if t != '\n']
    split_text = [[t for t in st if t != ''] for st in split_text]
    v_ids = [st[0] for st in split_text]
    v_scores = [list(map(float, st[1:])) for st in split_text]

    return v_ids, v_scores


def load_charades(gt_path):
    gt_ids = []
    gt_classes = []
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_ids.append(row['id'])
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [int(c[1:]) for c, _, _ in actions]
            gt_classes.append(actions)

    return gt_ids, gt_classes


def load_causal_charades(gt_path, data_path):
    gt_ids = []
    gt_classes = []
    with open(gt_path) as f:
        reader = csv.DictReader(f)
        for ir, row in enumerate(reader):
            vid = row['id']
            actions = row['actions']
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{
                    'class': x,
                    'start': float(y),
                    'end': float(z)
                } for x, y, z in actions]
            video_stream = VideoStream(os.path.join(data_path, vid+'.mp4'), actions)
            first_frame = video_stream.first_frame
            frame_target = video_stream.target['target']
            for f_id, frame_t in enumerate(frame_target):
                gt_ids.append('{}_{:06d}'.format(vid, first_frame + f_id))
                gt_classes.append(frame_t.numpy())

    return gt_ids, gt_classes


def save(log_file, gt_file, output_file, per_frame=False, calibrated=False,
         batch_time=None, data_time=None):
    mAP, wAP, ap = charades_v1_classify(log_file, gt_file, per_frame, calibrated)

    with open(output_file, 'w') as file:
        file.write('### {}MAP ### \n'.format('Calibrated ' if calibrated else ''))
        file.write('{:.02%}'.format(mAP))

        file.write('\n\n### {}wAP ### \n'.format('Calibrated ' if calibrated else ''))
        file.write('{:.02%}'.format(wAP))

        if batch_time and data_time:
            file.write('\n\n### Eval Time ###\n')
            file.write('Batch Time: {batch_time.avg:.3f}s avg. | '
                       'Data loading time: {data_time.avg:.3f}s avg.\n'.format(
                        batch_time=batch_time, data_time=data_time))

        file.write('\n\n{:5} | {:^5}\n'.format('class', '{}AP'.format('c' if calibrated else '')))
        for i, ap in enumerate(ap):
            file.write('{:5} | {:.02%}\n'.format(i, ap))


def get_w_array(targets):
    targets = np.array(targets, dtype='int')
    num_classes = targets.shape[1]
    pos_count = np.zeros(num_classes)
    neg_count = np.zeros(num_classes)

    sum_targets = targets.sum(axis=0)
    pos_count = sum_targets
    neg_count = (np.ones(num_classes)*len(targets) - sum_targets)

    return neg_count/pos_count
