import os
import torch
import torch.nn as nn

"""model_url: resnet18: https://download.pytorch.org/models/resnet18-5c106cde.pth"""
              
model_dir = os.path.expanduser('./weights')
model_name = 'resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        
        residual = x

        y_b1 = x.view(x.shape[0], -1)
        y_b1 = count_ones(y_b1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        y_b2 = out.view(out.shape[0], -1)
        y_b2 = count_ones(y_b2)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            y_b3 = x.view(x.shape[0], -1)
            y_b3 = count_ones(y_b3)
            residual = self.downsample(x)
        else:
            y_b3 = 0

        out += residual
        out = self.relu(out)
        
        y_b = y_b1 + y_b2 + y_b3

        return out, y_b


class ResNet18_stat(nn.Module):

    def __init__(self, block, layers, args=None, num_classes=1000):
        super().__init__()
        
        self.beta = args.beta
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        y0 = x.view(x.shape[0], -1)
        y0 = count_ones(y0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, y11 = self.layer1[0](x)
        x, y12 = self.layer1[1](x)
        x, y21 = self.layer2[0](x)
        x, y22 = self.layer2[1](x)
        x, y31 = self.layer3[0](x)
        x, y32 = self.layer3[1](x)
        x, y41 = self.layer4[0](x)
        x, y42 = self.layer4[1](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        y = y0 + y11 + y12 + y21 + y22 + y31 + y32 + y41 + y42

        return x, y

def count_ones(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return (activation_count - zeros)

def resnet18_stat(args=None):
    return ResNet18_stat(BasicBlock, [2, 2, 2, 2], args=args)
