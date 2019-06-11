import torch
import torch.nn as nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, addnon):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes,
                               kernel_size=(1 + temp_conv * 2, 1, 1),
                               padding=(temp_conv, 0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, stride, stride),
                               padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4,
                               kernel_size=(1, 1, 1),
                               padding=(0, 0, 0),
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.addnon = addnon
        if self.addnon:
            self.nonlocal_block = NonLocalBlock(in_channels=planes * 4)

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

    def __init__(self, block, layers, temp_conv, nonlocal_block, frame_num=32, num_classes=400,
                 non_local=True):
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
        self.maxpool = nn.MaxPool3d(kernel_size=(temp_stride, 3, 3),
                                    stride=(temp_stride, 2, 2),
                                    padding=(0, 0, 0))
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       temp_conv=temp_conv[0],
                                       nonlocal_block=nonlocal_block[0])
        self.temporalpool = nn.MaxPool3d(kernel_size=(2, 1, 1),
                                         stride=(2, 1, 1),
                                         padding=(0, 0, 0))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       temp_conv=temp_conv[1],
                                       nonlocal_block=nonlocal_block[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       temp_conv=temp_conv[2],
                                       nonlocal_block=nonlocal_block[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       temp_conv=temp_conv[3],
                                       nonlocal_block=nonlocal_block[3])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgdrop = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.mode = 'train'

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, temp_conv, nonlocal_block):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=(1, 1, 1), stride=(1, stride, stride),
                          padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, temp_conv=temp_conv[0],
                            addnon=nonlocal_block[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, downsample=None,
                                temp_conv=temp_conv[i],
                                addnon=nonlocal_block[i] if self.non_local else False))

        return nn.Sequential(*layers)

    def set_mode(self, mode):
        assert mode in ['train', 'val', 'test']

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
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

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
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))

            self.g = nn.Sequential(max_pool_layer, self.g)
            self.phi = nn.Sequential(max_pool_layer, self.phi)

    def forward(self, x):
        """
        Embedded gaussian operation in self-attention mode.

        Argument:
            x: (b, c, t, h, w)
        """
        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        f = torch.matmul(theta_x, phi_x)
        f_sc = f * (self.inter_channels**-.5)  # https://arxiv.org/pdf/1706.03762.pdf section 3.2.1
        f_div_C = F.softmax(f_sc, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


def resnet50(pretrained=False, **kwargs):
    temp_conv = [
        [1, 1, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0]
    ]
    nonlocal_block = [
        [0, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [0, 0, 0]
    ]
    model = I3DResNet(Bottleneck, [3, 4, 6, 3], temp_conv=temp_conv, nonlocal_block=nonlocal_block,
                      **kwargs)

    return model
