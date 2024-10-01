import torch
import torch.nn as nn

class VGG16_stat(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True, args=None):
        super().__init__()
        
        self.beta = args.beta
        
        self.features = nn.Sequential(
            
    	    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
    	    nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(

    	    nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
    
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
    
            nn.Linear(4096, num_classes))
        
        self.my_fc1 = nn.Linear(3*224*224, 1, bias=False)
        self.my_fc2 = nn.Linear(64*224*224, 1, bias=False)
        self.my_fc3 = nn.Linear(64*112*112, 1, bias=False)
        self.my_fc4 = nn.Linear(128*112*112, 1, bias=False)
        self.my_fc5 = nn.Linear(128*56*56, 1, bias=False)
        self.my_fc6 = nn.Linear(256*56*56, 1, bias=False)
        self.my_fc7 = nn.Linear(256*56*56, 1, bias=False)
        self.my_fc8 = nn.Linear(256*28*28, 1, bias=False)
        self.my_fc9 = nn.Linear(512*28*28, 1, bias=False)
        self.my_fc10 = nn.Linear(512*28*28, 1, bias=False)
        self.my_fc11 = nn.Linear(512*14*14, 1, bias=False)
        self.my_fc12 = nn.Linear(512*14*14, 1, bias=False)
        self.my_fc13 = nn.Linear(512*14*14, 1, bias=False)
        self.my_fc14 = nn.Linear(25088, 1, bias=False)
        self.my_fc15 = nn.Linear(4096, 1, bias=False)

        self.my_fc16 = nn.Linear(15, 1, bias=False)

    def forward(self, x):
        
        y1 = x.view(x.shape[0], -1)
        y1 = count_ones(y1)
        x = self.features[0](x)
        x = self.features[1](x)

        y2 = x.view(x.shape[0], -1)
        y2 = count_ones(y2)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)

        y3 = x.view(x.shape[0], -1)
        y3 = count_ones(y3)
        x = self.features[5](x)
        x = self.features[6](x)

        y4 = x.view(x.shape[0], -1)
        y4 = count_ones(y4)
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)

        y5 = x.view(x.shape[0], -1)
        y5 = count_ones(y5)
        x = self.features[10](x)
        x = self.features[11](x)

        y6 = x.view(x.shape[0], -1)
        y6 = count_ones(y6)
        x = self.features[12](x)
        x = self.features[13](x)

        y7 = x.view(x.shape[0], -1)
        y7 = count_ones(y7)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)

        y8 = x.view(x.shape[0], -1)
        y8 = count_ones(y8)
        x = self.features[17](x)
        x = self.features[18](x)

        y9 = x.view(x.shape[0], -1)
        y9 = count_ones(y9)
        x = self.features[19](x)
        x = self.features[20](x)

        y10 = x.view(x.shape[0], -1)
        y10 = count_ones(y10)
        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)

        y11 = x.view(x.shape[0], -1)
        y11 = count_ones(y11)
        x = self.features[24](x)
        x = self.features[25](x)

        y12 = x.view(x.shape[0], -1)
        y12 = count_ones(y12)
        x = self.features[26](x)
        x = self.features[27](x)

        y13 = x.view(x.shape[0], -1)
        y13 = count_ones(y13)
        x = self.features[28](x)
        x = self.features[29](x)
        x = self.features[30](x)

        x = x.view(x.size(0), -1)

        y14 = x.view(x.shape[0], -1)
        y14 = count_ones(y14)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)

        y15 = x.view(x.shape[0], -1)
        y15 = count_ones(y15)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)

        x = self.classifier[6](x)

        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15

        return x, y
    
def count_ones(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return (activation_count - zeros)

def vgg16_stat(args=None):
    return VGG16_stat(args=args)
