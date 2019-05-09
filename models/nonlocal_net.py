import pickle
import re

import torch
import torch.nn as nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, time_kernel=1, space_stride=1, downsample=None,
                 addnon=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               kernel_size=(time_kernel, 1, 1),
                               padding=(int((time_kernel-1)/2), 0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, space_stride, space_stride),
                               padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes,
                               planes * 4,
                               kernel_size=(1, 1, 1),
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.addnon = addnon
        if self.addnon:
            self.nonlocal_block = NonLocalBlock(in_channels=planes * 4, mode='embedded_gaussian')

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        if self.addnon:
            out = self.nonlocal_block(out)
        return out


class I3DResNet(nn.Module):

    def __init__(self, block, layers, frame_num=32, num_classes=400, non_local=True):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.inplanes = 64
        self.non_local = non_local
        super(I3DResNet, self).__init__()
        temp_stride = 2 if frame_num == 32 else 1
        self.conv1 = nn.Conv3d(3, 64,
                               kernel_size=(5, 7, 7),
                               stride=(temp_stride, 2, 2),
                               padding=(2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 0, 0))
        self.layer1 = self._make_layer_inflat(block, 64, layers[0], first_block=True)
        self.temporalpool = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.layer2 = self._make_layer_inflat(block, 128, layers[1], space_stride=2)
        self.layer3 = self._make_layer_inflat(block, 256, layers[2], space_stride=2)
        self.layer4 = self._make_layer_inflat(block, 512, layers[3], space_stride=2)
        self.avgpool = nn.AvgPool3d((int(frame_num/8), 7, 7))
        self.avgdrop = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.mode = 'train'

    def _make_layer_inflat(self, block, planes, blocks, space_stride=1, first_block=False):
        downsample = None
        if space_stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1, 1), stride=(1, space_stride, space_stride),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        if blocks % 2 == 0 or first_block:
            time_kernel = 3
        else:
            time_kernel = 1

        # Add first block
        layers.append(block(self.inplanes, planes, time_kernel, space_stride, downsample,
                            addnon=False))
        self.inplanes = planes * block.expansion

        if first_block:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, time_kernel))
        elif blocks % 2 == 0:
            time_kernel = 1
            add_nonlocal = True
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, time_kernel,
                                    addnon=add_nonlocal if self.non_local else False))
                time_kernel = (time_kernel + 2) % 4
                add_nonlocal = not add_nonlocal
        else:
            time_kernel = 3
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, time_kernel))
                time_kernel = (time_kernel + 2) % 4

        return nn.Sequential(*layers)

    def set_mode(self, mode):
        self.mode = mode

        if self.mode == 'test':
            self.set_fully_conv_test()

    def set_fully_conv_test(self):
        """ Transform the last fc layer in a conv1x1 layer """
        fc_weights = self.fc.state_dict()["weight"]
        conv1x1 = nn.Conv3d(fc_weights.size(1), fc_weights.size(0), 1)
        conv1x1.weight.data = fc_weights.view(fc_weights.size(0), fc_weights.size(1), 1, 1, 1)
        self.conv1x1 = conv1x1

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.temporalpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.mode == 'train':
            x = x.view(x.size(0), -1)
            x = self.avgdrop(x)
            x = self.fc(x)
        elif self.mode == 'val':
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        elif self.mode == 'test':
            x = self.conv1x1(x)
            x = x.mean(4).mean(3).mean(2)

        return x


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        self.mode = mode
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

        else:
            self.W = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if self.mode == "embedded_gaussian":
            self.operation_function = self._embedded_gaussian
        else:
            raise NotImplementedError('Non-local mode: {} not implemented!'.format(self.mode))

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool3d(kernel_size=(1, 2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool3d(kernel_size=(1, 2, 2)))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def convert_i3d_weights(weigths_file_path, new_model, save_model=False, new_model_name=None):
    """Expanded from https://github.com/Tushar-N/pytorch-resnet3d """

    data = pickle.load(open(weigths_file_path, 'rb'), encoding='latin')['blobs']
    data = {k: v for k, v in data.items() if 'momentum' not in k}

    downsample_pat = re.compile('res(.)_(.)_branch1_.*')
    conv_pat = re.compile('res(.)_(.)_branch2(.)_.*')
    nonlocal_pat = re.compile('nonlocal_conv(.)_(.)_*')
    m2num = dict(zip('abc', [1, 2, 3]))
    suffix_dict = {
        'b': 'bias', 'w': 'weight', 's': 'weight', 'rm': 'running_mean', 'riv': 'running_var'}
    nonlocal_dict = {'out': 'W.0', 'bn': 'W.1', 'phi': 'phi.0', 'g': 'g.0', 'theta': 'theta'}

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
    pre_trained_file = ("/media/v-pakova/New Volume1/OnlineActionRecognition/models/pre-trained/"
                        "non-local/i3d_baseline_8x8_IN_pretrain_400k.pkl")
    blank_resnet_i3d = resnet50(non_local=True, frame_num=8)
    resnet_i3d = convert_i3d_weights(pre_trained_file, blank_resnet_i3d, save_model=True,
                                     new_model_name="../../../models/pre-trained/"
                                     "resnet50_i3d_kinetics_8x8")

    print(resnet_i3d)
