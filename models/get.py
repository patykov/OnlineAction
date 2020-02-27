import logging
from importlib import import_module

import torch.nn


def get_model(arch, backbone, pretrained_weights, num_classes, non_local, frame_num,
              fullyConv=False, fine_tune=False, log_name='training'):

    model = getattr(import_module('models.baselines.' + arch), backbone)(
        num_classes=num_classes, non_local=non_local, frame_num=frame_num)

    strict = True
    if pretrained_weights:
        LOG = logging.getLogger(name=log_name)
        LOG.info('Loading pretrained-weights from {}'.format(pretrained_weights))
        weights = torch.load(pretrained_weights)
        if 'model' in weights:  # When loading from a checkpoint
            weights = weights['model']
        if fine_tune:
            LOG.info('Removing last layer weights to fine-tune')
            weights = {k: v for k, v in weights.items() if 'fc.' not in k}
            strict = False
        keys = model.load_state_dict(weights, strict=strict)
        LOG.info(keys)
    if fullyConv:
        model.set_fullyConv()
        LOG.info('Setting fully-convolutional inference')

    if torch.cuda.is_available():
        model.cuda()

    return model
