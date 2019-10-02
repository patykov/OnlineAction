import json
import os

import numpy as np

from metrics.metrics import get_accuracy
from metrics.charades_classify import charades_v1_classify
from metrics.kinetics_classify import read_file, get_top_predictions


def test_charades_map():
    abs_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(abs_dir, 'data', 'charades')

    log_file = os.path.join(data_dir, 'eval_results.txt')
    gt_file = os.path.join(data_dir, 'Charades_v1_test.csv')
    json_file = os.path.join(data_dir, 'classify_results.json')

    mAP, wAP, ap = charades_v1_classify(log_file, gt_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

    assert np.isclose(float(data['map']), mAP)

    assert np.isclose(float(data['wap']), wAP)

    assert np.allclose(np.array(data['ap_all'], dtype=float), ap, atol=1e-04)


def test_t1_t5_accuracy():
    abs_dir = os.path.split(os.path.abspath(__file__))[0]
    data_dir = os.path.join(abs_dir, 'data', 'kinetics')

    log_file = os.path.join(data_dir, 'eval_results.txt')

    labels, preditions, _ = read_file(log_file)

    top1_pred, top5_pred = get_top_predictions(preditions, labels)
    cls_acc1 = get_accuracy(top1_pred, labels)
    cls_acc5 = get_accuracy(top5_pred, labels)

    assert np.isclose(cls_acc1, 0.2)
    assert np.isclose(cls_acc5, 0.7)
