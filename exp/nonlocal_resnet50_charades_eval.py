# flake8: noqa

import sys

sys.path.insert(0, '..')
from eval_model import main


args = [
    '--map_file', '/data/Datasets/Charades/Annotations/Charades_v1_test.csv',
    '--root_data_path', '/data/Datasets/Charades/Charades_v1_480',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/' +
                            'charades_resnet50nl32_full_config1/' +
                            'charades_resnet50nl32_full_config1_best_model.pth',
    '--log_file', 'eval_charades_r50_i3d_nl_32x2_test',
    '--sample_frames', '32',
    '--dataset', 'charades',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net',
    '--mode', 'test'

]
sys.argv.extend(args)

main()
