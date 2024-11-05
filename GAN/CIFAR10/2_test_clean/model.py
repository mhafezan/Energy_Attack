import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, beta=15):
        super(AlexNet, self).__init__()

        self.beta  = beta

        self.features = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                 
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

        self.my_fc1 = nn.Linear(3*256*256, 1, bias=False)
        self.my_fc2 = nn.Linear(64*31*31, 1, bias=False)
        self.my_fc3 = nn.Linear(192*15*15, 1, bias=False)
        self.my_fc4 = nn.Linear(384*15*15, 1, bias=False)
        self.my_fc5 = nn.Linear(256*15*15, 1, bias=False)
        self.my_fc6 = nn.Linear(9216, 1, bias=False)
        self.my_fc7 = nn.Linear(4096, 1, bias=False)
        self.my_fc8 = nn.Linear(7, 1, bias=False)
        
    def forward(self, x):

        y1 = count_ones(x)
        x = self.features[0](x)  # conv
        x = self.features[1](x)  # relu        
        x = self.features[2](x)  # maxpool

        y2 = count_ones(x)
        x = self.features[3](x)  # conv
        x = self.features[4](x)  # relu        
        x = self.features[5](x)  # maxpool

        y3 = count_ones(x)
        x = self.features[6](x)  # conv
        x = self.features[7](x)  # relu

        y4 = count_ones(x)
        x = self.features[8](x)  # conv
        x = self.features[9](x)  # relu

        y5 = count_ones(x)
        x = self.features[10](x) # conv
        x = self.features[11](x) # relu        
        x = self.features[12](x) # maxpool     
        
        x = self.avgpool(x)

        x = x.view(x.size(0), 256*6*6)

        x = self.classifier[0](x) # drop

        y6 = count_ones(x)
        x = self.classifier[1](x) # linear
        x = self.classifier[2](x) # relu
        x = self.classifier[3](x) # drop

        y7 = count_ones(x)        
        x = self.classifier[4](x) # linear
        x = self.classifier[5](x) # relu
        x = self.classifier[6](x) # linear

        y = y1 + y2 + y3 + y4 + y5 + y6 + y7

        return x, y

def count_ones(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return (activation_count - zeros)

class Generator(nn.Module):
    def __init__(self, ngpu, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            
        return output.view(-1, 1).squeeze(1)