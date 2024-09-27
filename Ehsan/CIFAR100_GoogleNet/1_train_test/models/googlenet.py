"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn

class Inception(nn.Module):
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

      
        
    def forward(self, x):
        #return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

        
        s = self.b1[0](x)       #conv2
        s = self.b1[1](s)       #batch
        s = self.b1[2](s)       #relu

      
        t = self.b2[0](x)   #conv2
        t = self.b2[1](t)   #batch
        t = self.b2[2](t)   #relu

      
        t = self.b2[3](t)   #conv2
        t = self.b2[4](t)   #batch
        t = self.b2[5](t)   #relu

              
        
        u = self.b3[0](x)   #conv2
        u = self.b3[1](u)   #batch
        u = self.b3[2](u)   #relu

     
  
        u = self.b3[3](u)   #conv2
        u = self.b3[4](u)   #batch
        u = self.b3[5](u)   #relu

  
        u = self.b3[6](u)   #conv2
        u = self.b3[7](u)   #batch
        u = self.b3[8](u)   #relu      
 
        
        v = self.b4(x)      #relu: the last layer      


     
        return torch.cat([s, t, u, v], dim=1)

 
        

class GoogleNet(nn.Module):

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
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, 16, self.beta)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, 16, self.beta)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, 8, self.beta)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64, 8, self.beta)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, 8, self.beta)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, 8, self.beta)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, 8, self.beta)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, 4, self.beta)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, 4, self.beta)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

      

    def forward(self, x):
        # We break forward function into the minor layers: refer to https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/googlenet.py for the original function
        
        x = self.prelayer[0](x)  #conv2
        x = self.prelayer[1](x)  #batch
        x = self.prelayer[2](x)  #relu

      

        x = self.prelayer[3](x)  #conv2
        x = self.prelayer[4](x)  #batch
        x = self.prelayer[5](x)  #relu


        x = self.prelayer[6](x)  #conv2
        x = self.prelayer[7](x)  #batch
        x = self.prelayer[8](x)  #relu
      

        x = self.maxpool(x)

        x = self.a3(x)
        
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)

        x = self.b4(x)

        x = self.c4(x)
        
        x = self.d4(x)

        x  = self.e4(x)

        x = self.maxpool(x)

        x  = self.a5(x)

        x  = self.b5(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        
        
        return x

  


def googlenet(beta=15):
    return GoogleNet(beta=beta)
