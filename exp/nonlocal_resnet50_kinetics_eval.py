# flake8: noqa

import sys

sys.path.insert(0, '..')
from eval_model import main


args = [
    '--map_file', '/data/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--root_data_path', '/data/Datasets/Kinetics/400/val_clips_256',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_baseline_i3d_kinetics_32x2.pth',
    '--log_file', 'eval_baseline_kinetics_resnet50_32x2',
    '--sample_frames', '32',
    '--dataset', 'kinetics',
    '--mode', 'val',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net',
    '--baseline'

]
sys.argv.extend(args)

main()


# '/data/OnlineActionRecognition/outputs/' +
#                             'kinetics_resnet101_32_sec/kinetics_resnet101_32_sec.pth',