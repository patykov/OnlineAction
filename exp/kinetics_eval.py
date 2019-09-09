# flake8: noqa

import sys

sys.path.insert(0, '..')
from eval_model import main


args = [
    '--map_file', '/data/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--root_data_path', '/data/Datasets/Kinetics/400/val_clips_256',
    '--pretrained_weights', '/data/OnlineActionRecognition/outputs/' +
                            'kinetics_resnet101_32_sec/kinetics_resnet101_32_sec.pth',
    '--log_file', 'eval_kinetics_resnet101_32_sec',
    '--sample_frames', '32',
    '--dataset', 'kinetics',
    '--mode', 'test',
    '--backbone', 'resnet101'

]
sys.argv.extend(args)

main()
