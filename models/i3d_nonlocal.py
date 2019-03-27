import pickle
import re

import torch
import torch.nn as nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, time_kernel=1, space_stride=1,
                 downsample=None, addnon=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes,
                               planes,
                               kernel_size=(time_kernel, 1, 1),
                               padding=(int((time_kernel-1)/2), 0, 0),
                               bias=False)  # timepadding: make sure time-dim not reduce
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
            self.nonlocal_block = NonLocalBlock3D(in_channels=planes * 4, mode='embedded_gaussian')

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

    def __init__(self, block, layers, frame_num=32, num_classes=400):
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.inplanes = 64
        super(I3DResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64,
                               kernel_size=(5, 7, 7),
                               stride=(2, 2, 2),
                               padding=(2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.layer1 = self._make_layer_inflat(block, 64, layers[0], first_block=True)
        self.temporalpool = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.layer2 = self._make_layer_inflat(block, 128, layers[1], space_stride=2)
        self.layer3 = self._make_layer_inflat(block, 256, layers[2], space_stride=2)
        self.layer4 = self._make_layer_inflat(block, 512, layers[3], space_stride=2)
        self.avgpool = nn.AvgPool3d((int(frame_num/8), 7, 7))
        self.avgdrop = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                layers.append(block(self.inplanes, planes, time_kernel, addnon=add_nonlocal))
                time_kernel = (time_kernel + 2) % 4
                add_nonlocal = not add_nonlocal
        else:
            time_kernel = 3
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, time_kernel))
                time_kernel = (time_kernel + 2) % 4

        return nn.Sequential(*layers)

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
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(x.size(0), -1)
        x = self.avgdrop(x)
        x = self.fc(x)

        return x


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

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
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)

        else:
            self.W = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.kaiming_normal(self.W.weight)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if self.mode == "embedded_gaussian":
            self.operation_function = self._embedded_gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool3d(kernel_size=2))
            if self.phi is None:
                self.phi = nn.MaxPool3d(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, nn.MaxPool3d(kernel_size=2))

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


class NonLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(NonLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = I3DResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def copy_weigths_i3dResNet(weigths_file_path, new_model, save_model=False, new_model_name=None):
    with open(weigths_file_path, 'rb') as model_file:
        pretrained_data = pickle.load(model_file, encoding='latin1')

    pretrained_data1 = pretrained_data['blobs']
    # Removing training parameters
    pretrained_data2 = {k: v for k, v in pretrained_data1.items() if
                        ('momentum' not in k) and ('bn_rm' not in k) and ('bn_riv' not in k)
                        and ('lr' not in k) and ('model_iter' not in k)}

    pretrained_data_list = sorted(pretrained_data2.items())

    # Renaming layers
    pretrained_data3 = {}
    for k, v in pretrained_data_list:
        a = k[:]
        # Correcting the end
        if a[-2:] == '_b':
            a = a[:-2]+'.bias'
        elif a[-2:] == '_s' or a[-2:] == '_w':
            a = a[:-2]+'.weight'

        # Correcting the begin
        a = a.replace('res_conv1_bn', 'bn1')
        a = a.replace('pred', 'fc')
        r = re.compile('res._*')
        if r.match(a) is not None:
            layer = int(a[3])-1
            a = 'layer'+str(layer)+'.'+a[5:]

        # Correcting the middle
        a = a.replace('_branch1_bn', '.downsample.1')
        a = a.replace('_branch1', '.downsample.0')

        for i, l in enumerate(['a', 'b', 'c']):
            a = a.replace('_branch2{}_bn'.format(l), '.bn{}'.format(i+1))
            a = a.replace('_branch2{}'.format(l), '.conv{}'.format(i+1))

        # Correcting nonlocal
        if 'nonlocal' in a:
            layer = int(a[13])-1
            sub_layer = a[15]
            a = 'layer{}.{}.nonlocal_block.'.format(layer, sub_layer) + a[17:]
            a = a.replace('out', 'W.0')
            a = a.replace('bn', 'W.1')
            a = a.replace('phi', 'phi.0')
            a = a.replace('.g.', '.g.0.')

        pretrained_data3[a] = {'old_name': k, 'data': v}

    # Checking name, shape and number of layers
    param_i3d_keys = new_model.state_dict().keys()
    count = 0
    for k, v in pretrained_data3.items():
        assert(k in param_i3d_keys)
        count += 1
        assert(new_model.state_dict()[k].shape == v['data'].shape)

    assert count == len(list(new_model.named_parameters()))

    # Copying weigths
    for k, v in new_model.named_parameters():
        new_data = pretrained_data3[k]['data']
        v.data = nn.Parameter(torch.Tensor(new_data))

    if save_model:
        # Saving new model
        torch.save(new_model, '{}.pt'.format(new_model_name))

    return new_model


if __name__ == '__main__':
    pre_trained_file = ("/media/v-pakova/New Volume/OnlineActionRecognition/models/pre-trained/"
                        "non-local/i3d_nonlocal_32x2_IN_pretrain_400k.pkl")
    blank_resnet_i3d = resnet50()
    resnet_i3d = copy_weigths_i3dResNet(pre_trained_file, blank_resnet_i3d,
                                        save_model=True, new_model_name='testing_resnet50_i3d')

    print(resnet_i3d)
