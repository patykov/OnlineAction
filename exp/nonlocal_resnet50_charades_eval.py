# flake8: noqa

import sys

sys.path.insert(0, '..')
from eval_model import main


args = [
    '--map_file', '/data/Datasets/Charades/Annotations/Charades_v1_test.csv',
    '--root_data_path', '/data/Datasets/Charades/Charades_v1_480',
    '--pretrained_weights', '/data/OnlineActionRecognition/models/' +
                            'charades_resnet50nl32_mAP32/' +
                            'charades_resnet50nl32_config5_reducelr_full.pth',
    '--log_file', 'eval_charades_mAP32',
    '--sample_frames', '32',
    '--dataset', 'charades',
    '--backbone', 'resnet50',
    '--arch', 'nonlocal_net',
    '--mode', 'test'

]
sys.argv.extend(args)

main()
