"""google net in pytorch

[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn

class Inception_stat(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj, fc_in, beta=15):
        super().__init__()

        self.beta = beta
        
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

        #for computing sume of ones
        self.my_fc = nn.ModuleList([       
                       
            nn.Linear(input_channels*fc_in*fc_in, 1, bias=False) ,
            nn.Linear(n1x1*fc_in*fc_in, 1, bias=False) ,
            nn.Linear(n3x3_reduce*fc_in*fc_in, 1, bias=False) ,
            nn.Linear(n3x3*fc_in*fc_in, 1, bias=False) ,
            nn.Linear(n5x5_reduce*fc_in*fc_in, 1, bias=False) ,
            nn.Linear(n5x5*fc_in*fc_in, 1, bias=False) ,
            nn.Linear(input_channels*fc_in*fc_in, 1, bias=False) ,
            
            nn.Linear(7, 1, bias=False) 
        ])
      

        
    def forward(self, x):
        #return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

        #print(x.size())
        y1 = x.view(x.shape[0], -1)
        y1 = count_ones(y1)
        
        s = self.b1[0](x)       #conv2
        s = self.b1[1](s)       #batch
        s = self.b1[2](s)       #relu

        #print(s.size())
        y2 = s.view(s.shape[0], -1)
        y2 = count_ones(y2)

        t = self.b2[0](x)   #conv2
        t = self.b2[1](t)   #batch
        t = self.b2[2](t)   #relu

        #print(t.size())
        y3 = t.view(t.shape[0], -1)
        y3 = count_ones(y3)

        t = self.b2[3](t)   #conv2
        t = self.b2[4](t)   #batch
        t = self.b2[5](t)   #relu

        #print(t.size())
        y4 = t.view(t.shape[0], -1)
        y4 = count_ones(y4)
        
        u = self.b3[0](x)   #conv2
        u = self.b3[1](u)   #batch
        u = self.b3[2](u)   #relu

        #print(u.size())
        y5 = u.view(u.shape[0], -1)
        y5 = count_ones(y5)
  
        u = self.b3[3](u)   #conv2
        u = self.b3[4](u)   #batch
        u = self.b3[5](u)   #relu

        #print(u.size())
        y6 = u.view(u.shape[0], -1)
        y6 = count_ones(y6)

  
        u = self.b3[6](u)   #conv2
        u = self.b3[7](u)   #batch
        u = self.b3[8](u)   #relu
     
 
        
        v = self.b4[0](x)      #MaxPool2d

        #print(u.size())
        y7 = v.view(v.shape[0], -1)
        y7 = count_ones(y7)
        
        v = self.b4[1](v)      #conv2
        v = self.b4[2](v)      #batch
        v = self.b4[3](v)      #relu

       


        #y = self.my_fc[7]( torch.cat((y1,y2,y3,y4,y5,y6,y7),1) )
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7

        return torch.cat([s, t, u, v], dim=1), y
       
        

class GoogleNet_stat(nn.Module):

    def __init__(self, beta=15, num_class=100):
        super().__init__()

        self.beta = beta
        
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

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception_stat(192, 64, 96, 128, 16, 32, 32, 16, self.beta)
        self.b3 = Inception_stat(256, 128, 128, 192, 32, 96, 64, 16, self.beta)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception_stat(480, 192, 96, 208, 16, 48, 64, 8, self.beta)
        self.b4 = Inception_stat(512, 160, 112, 224, 24, 64, 64, 8, self.beta)
        self.c4 = Inception_stat(512, 128, 128, 256, 24, 64, 64, 8, self.beta)
        self.d4 = Inception_stat(512, 112, 144, 288, 32, 64, 64, 8, self.beta)
        self.e4 = Inception_stat(528, 256, 160, 320, 32, 128, 128, 8, self.beta)

        self.a5 = Inception_stat(832, 256, 160, 320, 32, 128, 128, 4, self.beta)
        self.b5 = Inception_stat(832, 384, 192, 384, 48, 128, 128, 4, self.beta)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)
       
        #for computing sume of ones
        self.my_fc_pl_0 = nn.Linear(3*32*32, 1, bias=False)
        self.my_fc_pl_1 = nn.Linear(64*32*32, 1, bias=False)
        self.my_fc_pl_2 = nn.Linear(64*32*32, 1, bias=False)

        self.my_fc_12 = nn.Linear(12, 1, bias=False)

        
    def forward(self, x):
        # We break forward function into the minor layers: refer to https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/googlenet.py for the original function
        
        #print(x.size())
        y1 = x.view(x.shape[0], -1)
        y1 = count_ones(y1)

        x = self.prelayer[0](x)  #conv2
        x = self.prelayer[1](x)  #batch
        x = self.prelayer[2](x)  #relu

        #print(x.size())
        y2 = x.view(x.shape[0], -1)
        y2 = count_ones(y2)

        x = self.prelayer[3](x)  #conv2
        x = self.prelayer[4](x)  #batch
        x = self.prelayer[5](x)  #relu

        #print(x.size())
        y3 = x.view(x.shape[0], -1)
        y3 = count_ones(y3)

        x = self.prelayer[6](x)  #conv2
        x = self.prelayer[7](x)  #batch
        x = self.prelayer[8](x)  #relu

      

        x = self.maxpool(x)

        x, a3_y = self.a3(x)
        
        x, b3_y = self.b3(x)

        x = self.maxpool(x)

        x, a4_y = self.a4(x)

        x, b4_y = self.b4(x)

        x, c4_y = self.c4(x)
        
        x, d4_y = self.d4(x)

        x, e4_y  = self.e4(x)

        x = self.maxpool(x)

        x, a5_y  = self.a5(x)

        x, b5_y  = self.b5(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        
        #y = self.my_fc_12(  torch.cat((y1, y2, y3, a3_y, b3_y, a4_y, b4_y, c4_y, d4_y, e4_y, a5_y, b5_y),1) )
        y = y1 + y2 + y3 + a3_y + b3_y + a4_y + b4_y + c4_y + d4_y + e4_y + a5_y + b5_y
        
        return x, y
     
 
 
def count_ones(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return (activation_count - zeros)

#974592, sum of all fc inputs
def googlenet_stat(beta=15):
    return GoogleNet_stat(beta=beta)
