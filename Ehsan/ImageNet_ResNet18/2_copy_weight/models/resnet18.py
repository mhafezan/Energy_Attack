import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, inplanes, planes, fc_in, stride=1, downsample=None, args=None):
        super().__init__()
        
        self.beta = args.beta
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        # To compute sum of ones
        if self.downsample is not None:
            self.my_fc = nn.ModuleList([
                nn.Linear((planes//stride)*fc_in*fc_in, 1, bias=False),
                nn.Linear(planes*(fc_in//stride)*(fc_in//stride), 1, bias=False),
                nn.Linear((planes//stride)*fc_in*fc_in, 1, bias=False),
                nn.Linear(3, 1, bias=False)])
        else:
            self.my_fc = nn.ModuleList([
                nn.Linear(planes*(fc_in//stride)*(fc_in//stride), 1, bias=False),
                nn.Linear(planes*(fc_in//stride)*(fc_in//stride), 1, bias=False),
                nn.Linear(2, 1, bias=False)])


    def forward(self, x):
        
        residual = x

        y1 = x.view(x.shape[0], -1)
        y1 = tanh(y1, self.beta)
        y1 = self.my_fc[0](y1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        y2 = out.view(out.shape[0], -1)
        y2 = tanh(y2, self.beta)
        y2 = self.my_fc[1](y2)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            y3 = x.view(x.shape[0], -1)
            y3 = tanh(y3, self.beta)
            y3 = self.my_fc[2](y3)
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        if self.downsample is not None:
            y = self.my_fc[3](torch.cat((y1,y2,y3), 1))
        else:
            y = self.my_fc[2](torch.cat((y1,y2), 1))

        return out, y
    
    def BasicBlock_Set_Weights_One(self):
        for i in range(len(self.my_fc)):
            self.my_fc[i].weight.data.fill_(1.)
        return


class ResNet18(nn.Module):

    def __init__(self, block, layers, args=None, num_classes=1000):
        super().__init__()
        
        self.beta = args.beta
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block,  64, layers[0], 56, stride=1, args=args)
        self.layer2 = self._make_layer(block, 128, layers[1], 56, stride=2, args=args)
        self.layer3 = self._make_layer(block, 256, layers[2], 28, stride=2, args=args)
        self.layer4 = self._make_layer(block, 512, layers[3], 14, stride=2, args=args)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # To compute number of ones
        self.my_fc_0  = nn.Linear(3*224*224, 1, bias=False)
        self.my_fc_9 = nn.Linear(9, 1, bias=False)

    def _make_layer(self, block, planes, blocks, fc_in, stride=1, args=None):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, fc_in, stride, downsample, args))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, fc_in=(fc_in//stride), args=args))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        y0 = x.view(x.shape[0], -1)
        y0 = tanh(y0, self.beta)
        y0 = self.my_fc_0(y0)
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
        
        y = self.my_fc_9(torch.cat((y0, y11, y12, y21, y22, y31, y32, y41, y42), 1))

        return x, y
    
    def resnet18_set_weights_one(self):

        self.my_fc_0.weight.data.fill_(1.)
        
        # Loop through each layer and call `BasicBlock_Set_Weights_One` for each BasicBlock
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                if isinstance(block, BasicBlock):
                    block.BasicBlock_Set_Weights_One()
        
        self.my_fc_9.weight.data.fill_(1.) 

def tanh(input_tensor, beta):
    output = torch.tanh(beta * input_tensor)
    output = torch.pow(output, 2)
    return output

def resnet18(args=None):
    return ResNet18(BasicBlock, [2, 2, 2, 2], args=args)