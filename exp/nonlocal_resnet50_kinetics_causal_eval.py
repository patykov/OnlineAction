# flake8: noqa

import sys

sys.path.insert(0, '..')
from causal_eval_model import main


args = [
    '--map_file', '/data/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--root_data_path', '/data/Datasets/Kinetics/400/val_clips_256',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_nonlocal_i3d_kinetics_32x2.pth',
    '--log_file', 'causal_eval_kinetics',
    '--sample_frames', '32',
    '--dataset', 'kinetics',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net',
    '--mode', 'val',
    '--subset'
]
sys.argv.extend(args)

main()