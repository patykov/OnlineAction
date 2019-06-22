import argparse
import pickle
import re

import torch

from nonlocal_net import resnet50


def convert_i3d_weights(weigths_file_path, new_model, save_model=False, new_model_name=None):
    """Expanded from https://github.com/Tushar-N/pytorch-resnet3d
    Convert pre-trained weights of I3DResnet from caffe2 to Pytorch.
    """

    data = pickle.load(open(weigths_file_path, 'rb'), encoding='latin')['blobs']
    data = {k: v for k, v in data.items() if 'momentum' not in k}

    downsample_pat = re.compile('res(.)_(.)_branch1_.*')
    conv_pat = re.compile('res(.)_(.)_branch2(.)_.*')
    nonlocal_pat = re.compile('nonlocal_conv(.)_(.)_*')
    m2num = dict(zip('abc', [1, 2, 3]))
    suffix_dict = {
        'b': 'bias', 'w': 'weight', 's': 'weight', 'rm': 'running_mean', 'riv': 'running_var'}
    nonlocal_dict = {'out': 'W.0', 'bn': 'W.1', 'phi': 'phi.1', 'g': 'g.1', 'theta': 'theta'}

    key_map = {'conv1.weight': 'conv1_w',
               'bn1.weight': 'res_conv1_bn_s',
               'bn1.bias': 'res_conv1_bn_b',
               'bn1.running_mean': 'res_conv1_bn_rm',
               'bn1.running_var': 'res_conv1_bn_riv',
               'fc.weight': 'pred_w',
               'fc.bias': 'pred_b'}

    for key in data:
        conv_match = conv_pat.match(key)
        if conv_match:
            layer, block, module = conv_match.groups()
            layer, block, module = int(layer), int(block), m2num[module]
            name = 'bn' if 'bn_' in key else 'conv'
            suffix = suffix_dict[key.split('_')[-1]]
            new_key = 'layer%d.%d.%s%d.%s' % (layer-1, block, name, module, suffix)
            key_map[new_key] = key

        ds_match = downsample_pat.match(key)
        if ds_match:
            layer, block = ds_match.groups()
            layer, block = int(layer), int(block)
            module = 0 if key[-1] == 'w' else 1
            name = 'downsample'
            suffix = suffix_dict[key.split('_')[-1]]
            new_key = 'layer%d.%d.%s.%d.%s' % (layer-1, block, name, module, suffix)
            key_map[new_key] = key

        nl_match = nonlocal_pat.match(key)
        if nl_match:
            layer, block = nl_match.groups()
            layer, block = int(layer), int(block)
            nl_op = nonlocal_dict[key.split('_')[-2]]
            suffix = suffix_dict[key.split('_')[-1]]
            name = 'nonlocal_block'
            new_key = 'layer%d.%d.%s.%s.%s' % (layer-1, block, name, nl_op, suffix)
            key_map[new_key] = key

    state_dict = new_model.state_dict()

    new_state_dict = {
        key: torch.from_numpy(data[key_map[key]]) for key in state_dict if key in key_map}

    # Check if weight dimensions match
    for key in state_dict:

        if key not in key_map:
            continue

        data_v, pth_v = data[key_map[key]], state_dict[key]
        assert str(tuple(data_v.shape)) == str(tuple(pth_v.shape)), (
            'Size Mismatch {} != {} in {}').format(data_v.shape, pth_v.shape, key)
        print('{:24s} --> {:40s} | {:21s}'.format(key_map[key], key, str(tuple(data_v.shape))))

    if save_model:
        # Saving new model weights
        torch.save(new_state_dict, '{}.pth'.format(new_model_name))

    return new_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='nonlocal')
    parser.add_argument('--frame_num', type=int, default=32)
    args = parser.parse_args()

    assert args.type in ['baseline', 'nonlocal'], ('Type %s not available. Choose between baseline '
                                                   'or nonlocal.' % args.type)
    assert args.frame_num in [8, 32], ('Number of frames %d not available. Choose between 8 or 32.'
                                       % args.frame_num)

    stride = 2 if args.frame_num == 32 else 8

    pre_trained_file = ('../../../models/pre-trained/non-local/'
                        'i3d_%s_%dx%d_IN_pretrain_400k.pkl' % (args.type, args.frame_num, stride))
    blank_resnet_i3d = resnet50(non_local=True if args.type == 'nonlocal' else False,
                                frame_num=args.frame_num)
    resnet_i3d = convert_i3d_weights(pre_trained_file, blank_resnet_i3d, save_model=True,
                                     new_model_name='../../../models/pre-trained/'
                                     'resnet50_%s_i3d_kinetics_%dx%d' % (args.type, args.frame_num,
                                                                         stride))

    print(resnet_i3d)
