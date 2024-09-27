import torch
import torch.nn as nn
import math
from utee import misc
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(inplanes, planes, stride)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(planes, planes)
        m['bn2'] = nn.BatchNorm2d(planes)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        self.zeros = [] # mohammad_sparsity_calculation
        self.sizes = [] # mohammad_sparsity_calculation
        
        if self.downsample is not None:
            residual = self.downsample(x)
            zz, ss = sparsity(residual) # mohammad_sparsity_calculation
            self.zeros.append(zz)       # mohammad_sparsity_calculation
            self.sizes.append(ss)       # mohammad_sparsity_calculation
        else:
            residual = x

        x = self.group1[0](x)           # mohammad_sparsity_calculation
        x = self.group1[1](x)           # mohammad_sparsity_calculation
        x = self.group1[2](x)           # mohammad_sparsity_calculation
        zz , ss = sparsity(x)           # mohammad_sparsity_calculation
        self.zeros.append(zz)           # mohammad_sparsity_calculation
        self.sizes.append(ss)           # mohammad_sparsity_calculation
        x = self.group1[3](x)           # mohammad_sparsity_calculation
        x = self.group1[4](x)           # mohammad_sparsity_calculation
        
        # mohammad: converted this x = self.group1(x) + residual to the following:
        x = x + residual
        x = self.relu(x)
        zz , ss = sparsity(x)           # mohammad_sparsity_calculation
        self.zeros.append(zz)           # mohammad_sparsity_calculation
        self.sizes.append(ss)           # mohammad_sparsity_calculation

        return x, self.zeros, self.sizes


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m['bn1'] = nn.BatchNorm2d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group1= nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(512 * block.expansion, num_classes))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)
        L1_zero, L1_size = sparsity(x) # mohammad_sparsity_calculation

        x, L2_L3_zeros, L2_L3_sizes = self.layer1[0](x) # mohammad_sparsity_calculation
        x, L4_L5_zeros, L4_L5_sizes = self.layer1[1](x) # mohammad_sparsity_calculation
        
        x, L6_L8_zeros,  L6_L8_sizes = self.layer2[0](x) # mohammad_sparsity_calculation
        x, L9_L10_zeros, L9_L10_sizes = self.layer2[1](x) # mohammad_sparsity_calculation
        
        x, L11_L13_zeros, L11_L13_sizes   = self.layer3[0](x) # mohammad_sparsity_calculation
        x, L14_L15_zeros, L14_L15_sizes = self.layer3[1](x) # mohammad_sparsity_calculation
        
        x, L16_L18_zeros, L16_L18_sizes = self.layer4[0](x) # mohammad_sparsity_calculation
        x, L19_L20_zeros, L19_L20_sizes = self.layer4[1](x) # mohammad_sparsity_calculation

        x = self.avgpool(x)
        L21_zero, L21_size = sparsity(x) # mohammad_sparsity_calculation
        
        x = x.view(x.size(0), -1)
        
        x = self.group2(x)
        L22_zero, L22_size = sparsity(x) # mohammad_sparsity_calculation
        
        # mohammad: merge all layers into one single array
        L1_L22_zeros = [L1_zero] + L2_L3_zeros + L4_L5_zeros + L6_L8_zeros + L9_L10_zeros + L11_L13_zeros + L14_L15_zeros + L16_L18_zeros + L19_L20_zeros + [L21_zero] + [L22_zero]
        L1_L22_sizes = [L1_size] + L2_L3_sizes + L4_L5_sizes + L6_L8_sizes + L9_L10_sizes + L11_L13_sizes + L14_L15_sizes + L16_L18_sizes + L19_L20_sizes + [L21_size] + [L22_size]

        return x, L1_L22_zeros, L1_L22_sizes 


def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet18'], model_root)
    return model

# mohammad: To compute number of zeroes and activations in a tensor
def sparsity(y):
    zeros = torch.count_nonzero(torch.eq(y, 0)).item()
    activation_count = y.numel()
    return zeros, activation_count