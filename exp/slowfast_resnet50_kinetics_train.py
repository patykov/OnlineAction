# flake8: noqa

import sys

sys.path.insert(0, '..')
from train_model import main


args = [
    '--config_file',  'slowfast_kinetics_config.json',
    '--train_map_file', '/data/Datasets/Kinetics/400/Annotation/train_clips_256_list.txt',
    '--val_map_file', '/data/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--train_data_path', '/data/Datasets/Kinetics/400/train_clips_256',
    '--val_data_path', '/data/Datasets/Kinetics/400/val_clips_256',
    '--filename', 'slowfast_kinetics_train',
    '--sample_frames', '64',
    '--dataset', 'kinetics',
    '--backbone', 'resnet50',
    '--arch', 'slowfast_net',
    '--subset'

]
sys.argv.extend(args)

main()
