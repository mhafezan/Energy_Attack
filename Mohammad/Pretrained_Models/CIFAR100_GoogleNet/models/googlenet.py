import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        self.zeros = []
        self.sizes = []
        
        s = self.b1(x)
        zz, ss = sparsity(s)
        self.zeros.append(zz)      
        self.sizes.append(ss)      
        
        t = self.b2[0](x)
        t = self.b2[1](t)
        t = self.b2[2](t)
        zz, ss = sparsity(t)       
        self.zeros.append(zz)      
        self.sizes.append(ss)      
        t = self.b2[3](t)
        t = self.b2[4](t)
        t = self.b2[5](t)
        zz, ss = sparsity(t)       
        self.zeros.append(zz)      
        self.sizes.append(ss)      
        
        u = self.b3[0](x)
        u = self.b3[1](u)
        u = self.b3[2](u)
        zz, ss = sparsity(u)       
        self.zeros.append(zz)      
        self.sizes.append(ss)      
        u = self.b3[3](u)
        u = self.b3[4](u)
        u = self.b3[5](u)
        zz, ss = sparsity(u)       
        self.zeros.append(zz)      
        self.sizes.append(ss)      
        u = self.b3[6](u)
        u = self.b3[7](u)
        u = self.b3[8](u)
        zz, ss = sparsity(u)       
        self.zeros.append(zz)      
        self.sizes.append(ss)      
        
        v = self.b4(x)
        zz, ss = sparsity(v)       
        self.zeros.append(zz)      
        self.sizes.append(ss)      
        
        return torch.cat([s, t, u, v], dim=1), self.zeros, self.sizes


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        # Although we only use 1 conv layer as prelayer, we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        # In general, an Inception network is a network consisting of modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the grid
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x):
        # We break forward function into the minor layers: refer to https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/googlenet.py for the original function
        
        x = self.prelayer[0](x)
        x = self.prelayer[1](x)
        x = self.prelayer[2](x)
        L1_zeros, L1_sizes = sparsity(x)
        x = self.prelayer[3](x)
        x = self.prelayer[4](x)
        x = self.prelayer[5](x)
        L2_zeros, L2_sizes = sparsity(x)
        x = self.prelayer[6](x)
        x = self.prelayer[7](x)
        x = self.prelayer[8](x)
        L3_zeros, L3_sizes = sparsity(x)

        x = self.maxpool(x)
        L4_zeros, L4_sizes = sparsity(x)

        x, L5_L11_zeros, L5_L11_sizes = self.a3(x)
        
        x, L12_L18_zeros, L12_L18_sizes = self.b3(x)

        x = self.maxpool(x)
        L19_zeros, L19_sizes = sparsity(x)

        x, L20_L26_zeros, L20_L26_sizes = self.a4(x)

        x, L27_L33_zeros, L27_L33_sizes = self.b4(x)

        x, L34_L40_zeros, L34_L40_sizes = self.c4(x)
        
        x, L41_L47_zeros, L41_L47_sizes = self.d4(x)

        x, L48_L54_zeros, L48_L54_sizes  = self.e4(x)

        x = self.maxpool(x)
        L55_zeros, L55_sizes = sparsity(x)

        x, L56_L62_zeros, L56_L62_sizes  = self.a5(x)

        x, L63_69_zeros, L63_69_sizes  = self.b5(x)

        x = self.avgpool(x)
        L70_zeros, L70_sizes = sparsity(x)
        x = self.dropout(x)
        L71_zeros, L71_sizes = sparsity(x)
        
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        
        L1_L71_zeros = [L1_zeros] + [L2_zeros] + [L3_zeros] + [L4_zeros] + L5_L11_zeros + L12_L18_zeros + [L19_zeros] + L20_L26_zeros + L27_L33_zeros + L34_L40_zeros + L41_L47_zeros + L48_L54_zeros + [L55_zeros] + L56_L62_zeros + L63_69_zeros + [L70_zeros] + [L71_zeros]
        L1_L71_sizes = [L1_sizes] + [L2_sizes] + [L3_sizes] + [L4_sizes] + L5_L11_sizes + L12_L18_sizes + [L19_sizes] + L20_L26_sizes + L27_L33_sizes + L34_L40_sizes + L41_L47_sizes + L48_L54_sizes + [L55_sizes] + L56_L62_sizes + L63_69_sizes + [L70_sizes] + [L71_sizes]

        return x, L1_L71_zeros, L1_L71_sizes

def sparsity(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

def googlenet():
    return GoogleNet()