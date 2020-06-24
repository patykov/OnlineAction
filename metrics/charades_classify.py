import csv
import re

import numpy as np


def charades_v1_classify(cls_file, gt_path, per_frame=False):
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
    gt_ids, gt_classes = read_file(gt_path) if per_frame else load_charades(gt_path)
    n_test = len(gt_ids)

    # Load test scores
    test_ids, test_scores = read_file(cls_file)

    # Check if there are duplicate items (caused by the parallel execution)
    test_ids, test_index_order = np.unique(test_ids, return_index=True)
    test_scores = np.array(test_scores)[test_index_order]

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
        raise RuntimeError('There are {} extra items!'.format(n-n_test))

    mAP, wAP, ap, _, _, cAP, _, _ = charades_map(np.array(test_scores), np.array(gt_classes))

    return mAP, wAP, ap, cAP


def map_func(submission_array, gt_array):
    """ Returns mAP, weighted mAP, AP array, precisions, recall and calibrated AP"""
    m_aps = []
    c_aps = []
    a_prec = np.zeros(submission_array.shape)
    a_recall = np.zeros(submission_array.shape)
    n_samples = submission_array.shape[0]
    n_classes = submission_array.shape[1]
    new_preds = submission_array.copy()
    for oc_i in range(n_classes):
        sorted_idxs = np.argsort(-submission_array[:, oc_i])
        new_preds[:, oc_i] = new_preds[:, oc_i][sorted_idxs]
        sorted_gt = gt_array[:, oc_i][sorted_idxs]
        tp = sorted_gt == 1
        fp = np.invert(tp)
        n_pos = tp.sum()
        n_gt = sorted_gt.sum()

        t_pcs = np.cumsum(tp)
        f_pcs = np.cumsum(fp)
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        recall = t_pcs / n_gt.astype(float)

        # Calibrated prec
        w = (n_samples - n_gt) / float(n_gt)
        c_t_pcs = t_pcs * w
        c_prec = c_t_pcs / (f_pcs + c_t_pcs).astype(float)

        avg_prec = 0
        c_avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
                c_avg_prec += c_prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
        c_aps.append(c_avg_prec / n_pos.astype(float))
        a_prec[:, oc_i] = prec
        a_recall[:, oc_i] = recall
    m_aps = np.array(m_aps)
    c_aps = np.array(c_aps)
    m_ap = np.nanmean(m_aps)
    c_ap = np.nanmean(c_aps)
    w_ap = np.nansum(m_aps * gt_array.sum(axis=0) / gt_array.sum().astype(float))
    return m_ap, w_ap, m_aps, a_prec, a_recall, c_ap, c_aps, new_preds


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """

    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF

    return map_func(fix, gt_array)


def get_thresholds(test_scores, gt_classes):
    _, _, ap, prec, _, _, _, _ = charades_map(test_scores, gt_classes)

    n_classes = test_scores.shape[1]
    thresholds = np.zeros(n_classes)
    for c in range(n_classes):
        idx = np.where(prec[:, c] > ap[c])[0][-1]
        thresholds[c] = test_scores[idx, c]
    return thresholds


def read_file(file_path):
    with open(file_path, 'r') as file:
        text = sorted(file.readlines())

    split_text = [t.strip().split(' ') for t in text]
    v_ids = [st[0] for st in split_text]
    v_scores = [list(map(float, st[1:])) for st in split_text]

    return v_ids, v_scores


def read_causal_file(file_path):
    text = ''
    with open(file_path, 'r', encoding="ISO-8859-1") as file:
        for line in file:
            text += ' ' + line.strip()

    lines_text = [t for t in text.split(' ') if t != '']

    # Getting number of 'items' per line
    num_per_line = 1  # counting the first label
    for value in lines_text[1:]:
        if '_' in value:
            # Break when next label is found
            break
        num_per_line += 1

    split_text = [
        lines_text[i-num_per_line:i] for i in range(num_per_line, len(lines_text), num_per_line)]

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


def save(log_file, gt_file, output_file, per_frame=False, calibrated=False,
         batch_time=None, data_time=None):
    mAP, wAP, ap, _ = charades_v1_classify(log_file, gt_file, per_frame)

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


def get_gt_per_verb(gt_array, verb_classes_file):
    class_to_verb = {}
    with open(verb_classes_file, 'r') as file:
        for line in file:
            class_match = re.match(r'(\d*) (.*) (\d*)', line)
            if class_match:
                old_class_id, new_class_id, _ = class_match.groups()
            class_to_verb[int(old_class_id)] = int(new_class_id)

    num_classes = len(np.unique(list(class_to_verb.values())))
    new_gt_array = np.zeros((gt_array.shape[0], num_classes))
    print(new_gt_array.shape)
    for frame_id, frame_gt in enumerate(gt_array):
        old_classes = np.where(frame_gt)[0]
        new_classes = [class_to_verb[c_id] for c_id in old_classes]
        new_gt_array[frame_id, new_classes] = 1

    return new_gt_array
