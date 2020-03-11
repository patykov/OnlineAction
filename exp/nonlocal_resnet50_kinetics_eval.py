# flake8: noqa

import sys

sys.path.insert(0, '..')
from eval_model import main


args = [
    '--map_file', '/data/Datasets/Kinetics/400/Annotation/val_clips_256_list.txt',
    '--root_data_path', '/data/Datasets/Kinetics/400/val_clips_256',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/pre-trained/' +
                            'resnet50_nonlocal_i3d_kinetics_8x8.pth',
    '--log_file', 'eval_nonlocal_kinetics_resnet50_8x8_video_centerCrop',
    '--sample_frames', '8',
    '--dataset', 'kinetics',
    '--mode', 'video_centerCrop',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net'

]
sys.argv.extend(args)

main()
