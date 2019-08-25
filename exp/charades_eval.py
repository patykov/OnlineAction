# flake8: noqa

import sys

sys.path.insert(0, '..')
from eval_model import main


args = [
    '--map_file', '/media/v-pakova/Datasets/Charades/Annotations/Charades_v1_test.csv',
    '--root_data_path', '/media/v-pakova/Datasets/Charades/Charades_v1_480',
    '--pretrained_weights', '/media/v-pakova/OnlineActionRecognition/models/outputs/' +
                            'charades_nl_finetune_32x2_config3_full/' +
                            'charades_nl_finetune_32x2_config3_full.pth',
    '--log_file', 'eval_charades_config3_full_test1',
    '--sample_frames', '32',
    '--dataset', 'charades',
    '--mode', 'test'

]
sys.argv.extend(args)

main()
