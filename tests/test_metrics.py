import json
import os

import numpy as np

from metric_tools.charades_classify import charades_v1_classify


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
