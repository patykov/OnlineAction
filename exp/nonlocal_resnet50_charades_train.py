# flake8: noqa

import sys

sys.path.insert(0, '..')
from train_model import main


args = [
    '--config_file',  'charades_baseline1.json',
    '--train_map_file', '/data/Datasets/Charades/Annotations/Charades_v1_train.csv',
    '--val_map_file', '/data/Datasets/Charades/Annotations/Charades_v1_test.csv',
    '--train_data_path', '/data/Datasets/Charades/Charades_v1_480',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_baseline_i3d_kinetics_32x2.pth',
    '--pos_weights', '/data/Datasets/Charades/charades_pos_weight.pt',
    '--filename', 'charades_r50baseline_32x2',
    '--sample_frames', '32',
    '--dataset', 'charades',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net',
    '--fine_tune',
    '--restart',
    '--subset'

]
sys.argv.extend(args)

main()
