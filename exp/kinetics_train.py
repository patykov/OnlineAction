# flake8: noqa

import sys

sys.path.insert(0, '..')
from train_model import main


args = [
    '--config_file',  'kinetics_config.json',
    '--train_map_file', '/media/v-pakova/Datasets/Kinetics/400/Annotation/train_clips_256_list.txt',
    '--val_map_file', '/media/v-pakova/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--train_data_path', '/media/v-pakova/Datasets/Kinetics/400/train_clips_256',
    '--val_data_path', '/media/v-pakova/Datasets/Kinetics/400/val_clips_256',
    '--pretrained_weights', '/media/v-pakova/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_nonlocal_i3d_kinetics_32x2.pth',
    '--filename', 'kinetics_train',
    '--sample_frames', '32',
    '--dataset', 'kinetics',
    '--subset'

]
sys.argv.extend(args)

main()
