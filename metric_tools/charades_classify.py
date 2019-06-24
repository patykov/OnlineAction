import csv

import numpy as np


def charades_v1_classify(cls_file, gt_path):
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
    gt_ids, gt_classes = load_charades(gt_path)
    n_classes = 157
    n_test = len(gt_ids)

    # Load test scores
    test_ids, test_scores = read_file(cls_file)
    n = len(test_scores)
    if n < n_test:
        print('Warning: {} Videos missing\n'.format(n_test-n))
    elif n_test < n:
        print('Warning: {} Extra videos\n'.format(n-n_test))

    predictions = {}
    for i, vid in enumerate(test_ids):
        predictions[vid] = test_scores[i]

    # Compare test scores to ground truth
    gtlabel = np.zeros((n_test, n_classes))
    test = np.zeros((n_test, n_classes))
    for i, vid in enumerate(gt_ids):
        gtlabel[i, gt_classes[i]] = 1
        if vid in predictions:  # For partial evaluation
            test[i, :] = predictions[vid]

    mAP, wAP, ap = charades_map(test, gtlabel)

    print('MAP: {:.5f}\n'.format(mAP))

    return mAP, wAP, ap


def map_func(submission_array, gt_array):
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
        prec = t_pcs / (f_pcs+t_pcs).astype(float)
        avg_prec = 0
        for i in range(submission_array.shape[0]):
            if tp[i]:
                avg_prec += prec[i]
        m_aps.append(avg_prec / n_pos.astype(float))
    m_aps = np.array(m_aps)
    m_ap = np.mean(m_aps)
    w_ap = (m_aps * gt_array.sum(axis=0) / gt_array.sum().astype(float))
    return m_ap, w_ap, m_aps


def charades_map(submission_array, gt_array):
    """
    Approximate version of the charades evaluation function
    For precise numbers, use the submission file with the official matlab script
    """

    fix = submission_array.copy()
    empty = np.sum(gt_array, axis=1) == 0
    fix[empty, :] = np.NINF

    return map_func(fix, gt_array)


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


def save(log_file, gt_file, output_file, batch_time, data_time):
    mAP, wAP, ap = charades_v1_classify(log_file, gt_file)

    with open(output_file, 'w') as file:
        file.write('### MAP ### \n')
        file.write('{:.5f}'.format(mAP))

        if batch_time and data_time:
            file.write('\n\n### Eval Time ###\n')
            file.write('Batch Time: {batch_time.avg:.3f}s avg. | '
                       'Data loading time: {data_time.avg:.3f}s avg.\n'.format(
                        batch_time=batch_time, data_time=data_time))

        file.write('\n\n{:5} | {:10} | {:10}\n'.format(
            'class', 'avg. prec.', 'weighted avg. prec.'))
        for i, (ap, wap) in enumerate(zip(ap, wAP)):
            file.write('{:5} | {:10.5f} | {:10.5f}\n'.format(i, ap, wap))
