# flake8: noqa

import sys

sys.path.insert(0, '..')
from causal_eval_model import main


args = [
    '--map_file', '/data/Datasets/Charades/Annotations/Charades_v1_test.csv',
    '--root_data_path', '/data/Datasets/Charades/Charades_v1_480',
    '--pretrained_weights', '/data/OnlineActionRecognition/outputs/' +
                            'charades_resnet50nl32_full_classes_by_verb/' +
                            'charades_resnet50nl32_full_classes_by_verb_best_model.pth',
    '--log_file', 'causal_eval_charades_resnet50nl32_full_classes_by_verb',
    '--sample_frames', '32',
    '--dataset', 'charades',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net',
    '--mode', 'val',
    '--verb_classes_file', '/data/Datasets/Charades/Annotations/Charades_v1_classes_by_verb.txt',
    '--subset'
]
sys.argv.extend(args)

main()
