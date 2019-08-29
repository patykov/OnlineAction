# flake8: noqa

import sys

sys.path.insert(0, '..')
from eval_model import main


args = [
    '--map_file', '/media/v-pakova/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--root_data_path', '/media/v-pakova/Datasets/Kinetics/400/val_clips_256',
    '--pretrained_weights', '/media/v-pakova/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_nonlocal_i3d_kinetics_32x2.pth',
    '--log_file', 'eval_kinetics',
    '--sample_frames', '32',
    '--dataset', 'kinetics',
    '--mode', 'test'

]
sys.argv.extend(args)

main()
