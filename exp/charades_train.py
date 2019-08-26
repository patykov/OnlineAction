# flake8: noqa

import sys

sys.path.insert(0, '..')
from train_model import main


args = [
    '--config_file',  'config3.json',
    '--train_map_file', '/media/v-pakova/Datasets/Charades/Annotations/Charades_v1_train.csv',
    '--val_map_file', '/media/v-pakova/Datasets/Charades/Annotations/Charades_v1_test.csv',
    '--train_data_path', '/media/v-pakova/Datasets/Charades/Charades_v1_480',
    '--pretrained_weights', '/media/v-pakova/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_nonlocal_i3d_kinetics_32x2.pth',
    '--pos_weights', '/media/v-pakova/Datasets/Charades/charades_pos_weight.pt',
    '--filename', 'charades_test',
    '--sample_frames', '32',
    '--dataset', 'charades',
    '--fine_tune',
    '--restart',
    '--subset'

]
sys.argv.extend(args)

main()
