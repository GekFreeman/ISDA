from collections import OrderedDict
import pdb
import torch.nn as nn

from .encoders import register
from ..modules import *

__all__ = ['resnet50', 'wide_resnet50']


def conv3x3(in_channels, out_channels, stride=1):
    return Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False)


class Block(Module):
    expansion = 1
    def __init__(self, in_planes, planes, bn_args, stride=1, downsample=None):
        super(Block, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = BatchNorm2d(planes, **bn_args)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, **bn_args)

        self.downsample = downsample
        if stride > 1:
            self.res_conv = Sequential(
                OrderedDict([
                    ('conv', conv1x1(in_planes, planes, stride)),
                    ('bn', BatchNorm2d(planes, **bn_args)),
                ]))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, params=None, episode=None):
        y = x
        out = self.conv1(x, get_child_dict(params, 'conv1'))
        out = self.bn1(out, get_child_dict(params, 'bn1'), episode)
        out = self.relu(out)
        out = self.conv2(out, get_child_dict(params, 'conv2'))
        out = self.bn2(out, get_child_dict(params, 'bn2'), episode)
        if self.stride > 1:
            x = self.res_conv(x, get_child_dict(params, 'res_conv'), episode)
        
        out = self.relu(out + x)
        return out

class BottleNeck(Module):
    expansion = 4
    def __init__(self, in_planes, planes, bn_args, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = BatchNorm2d(planes, **bn_args)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes, **bn_args)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion, **bn_args)

        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, params=None, episode=None):
        identity = x
        out = self.conv1(x, get_child_dict(params, 'conv1'))
        out = self.bn1(out, get_child_dict(params, 'bn1'), episode)
        out4 = self.relu(out)
        out4 = self.conv2(out4, get_child_dict(params, 'conv2'))
        out4 = self.bn2(out4, get_child_dict(params, 'bn2'), episode)
        out3 = self.relu(out4)
        out3 = self.conv3(out3, get_child_dict(params, 'conv3'))
        out3 = self.bn3(out3, get_child_dict(params, 'bn3'), episode)
        out2 = self.relu(out3)
        if self.downsample is not None:
            identity = self.downsample.conv1(x, get_child_dict(params, 'downsample.conv1'), episode)
            identity = self.downsample.bn1(identity, get_child_dict(params, 'downsample.bn1'), episode)
        
        out1 = self.relu(out2 + identity)
        return out1

class MLBlock(Module):
    def __init__(self, block, inplanes, planes, blocks, bn_args, stride=1):
        super(MLBlock, self).__init__()
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(OrderedDict([
                ('conv1', conv1x1(inplanes, planes * block.expansion, stride)),
                ('bn1', BatchNorm2d(planes * block.expansion))
            ]))
        self.block_dict = OrderedDict()
        self.block_dict['block0'] = block(inplanes, planes, bn_args, stride, downsample)
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            self.block_dict[f'block{i}'] = block(inplanes, planes, bn_args)
        self.blocks = nn.Sequential(self.block_dict)

    def forward(self, x, params=None, episode=None):
        for i,layer in enumerate(self.blocks):
            x = layer(x, get_child_dict(params, f"blocks.block{i}"), episode)
        return x
    

class ResNet50(Module):
    def __init__(self, block, layers, channels, bn_args):
        super(ResNet50, self).__init__()
        self.channels = channels
        self.inplanes = 64
        self.scale = block.expansion
        episodic = bn_args.get('episodic') or []
        bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
        bn_args_ep['episodic'] = True
        bn_args_no_ep['episodic'] = False
        bn_args_dict = dict()
        for i in [0, 1, 2, 3, 4]:
            if 'layer%d' % i in episodic:
                bn_args_dict[i] = bn_args_ep
            else:
                bn_args_dict[i] = bn_args_no_ep

        self.layer0 = Sequential(
            OrderedDict([
                ('conv', conv3x3(3, 64)),
                ('bn', BatchNorm2d(64, **bn_args_dict[0])),
            ]))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = MLBlock(block, 64, channels[0], layers[0], bn_args_dict[1])
        self.layer2 = MLBlock(block, channels[0] * block.expansion, channels[1], layers[1], bn_args_dict[2], stride=2)
        self.layer3 = MLBlock(block, channels[1] * block.expansion, channels[2], layers[2], bn_args_dict[3], stride=2)
        self.layer4 = MLBlock(block, channels[2] * block.expansion, channels[3], layers[3], bn_args_dict[4], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)
                
    def get_out_dim(self, scale=1):
        return self.out_dim * self.scale

    def forward(self, x, params=None, episode=None):
        out = self.layer0(x, get_child_dict(params, 'layer0'), episode)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out, get_child_dict(params, 'layer1'), episode)
        out = self.layer2(out, get_child_dict(params, 'layer2'), episode)
        out = self.layer3(out, get_child_dict(params, 'layer3'), episode)
        out = self.layer4(out, get_child_dict(params, 'layer4'), episode)
        out = self.pool(out).flatten(1)
        return out


@register('resnet50')
def resnet50(bn_args=dict()):
    return ResNet50(BottleNeck, [3, 4, 6, 3], [64, 128, 256, 512], bn_args)


@register('wide-resnet50')
def wide_resnet50(bn_args=dict()):
    return ResNet50(BottleNeck, [3, 4, 6, 3], [64, 160, 320, 640], bn_args)