# flake8: noqa

import sys

sys.path.insert(0, '..')
from train_model import main


args = [
    '--config_file',  'nonlocal_kinetics_config.json',
    '--train_map_file', '/data/Datasets/Kinetics/400/Annotation/train_clips_256_list.txt',
    '--val_map_file', '/data/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--train_data_path', '/data/Datasets/Kinetics/400/train_clips_256',
    '--val_data_path', '/data/Datasets/Kinetics/400/val_clips_256',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/pre-trained/' +
                            'resnet101_nonlocal_i3d_kinetics_32x2_partial.pth',
    '--filename', 'kinetics_train',
    '--sample_frames', '32',
    '--dataset', 'kinetics',
    '--backbone', 'resnet101',
    '--arch', 'nonlocal_net',
    '--subset'

]
sys.argv.extend(args)

main()
