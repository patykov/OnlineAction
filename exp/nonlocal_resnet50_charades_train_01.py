# flake8: noqa

import sys

sys.path.insert(0, '..')
from train_model import main


args = [
    '--config_file',  'nonlocal_charades_config.json',
    '--train_map_file', '/data/Datasets/Charades/Annotations/Charades_v1_train.csv',
    '--val_map_file', '/data/Datasets/Charades/Annotations/Charades_v1_test.csv',
    '--train_data_path', '/data/Datasets/Charades/Charades_v1_480',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_nonlocal_i3d_kinetics_32x2.pth',
    '--pos_weights', '/data/Datasets/Charades/charades_pos_weight.pt',
    '--filename', 'charades_test',
    '--sample_frames', '32',
    '--dataset', 'charades',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net',
    '--fine_tune',
    '--selected_classes_file', '/data/Datasets/Charades/Annotations/Charades_v1_classes_over_01.txt'

]
sys.argv.extend(args)

main()
