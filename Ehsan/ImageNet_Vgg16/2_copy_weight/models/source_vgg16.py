import torch
import torch.nn as nn

class VGG16_stat(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True, args=None):
        super(VGG16_stat, self).__init__()
        
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

    def forward(self, x):
        
        y1 = x.view(x.shape[0], -1)
        y1 = count_ones(y1)
        x = self.features[0](x)      #conv
        x = self.features[1](x)      #ReLU

        y2 = x.view(x.shape[0], -1)
        y2 = count_ones(y2)
        x = self.features[2](x)      #conv
        x = self.features[3](x)      #ReLU
        x = self.features[4](x)      #MaxPool2d

        y3 = x.view(x.shape[0], -1)
        y3 = count_ones(y3)
        x = self.features[5](x)      #conv
        x = self.features[6](x)      #ReLU

        y4 = x.view(x.shape[0], -1)
        y4 = count_ones(y4)
        x = self.features[7](x)      #conv
        x = self.features[8](x)      #ReLU
        x = self.features[9](x)      #MaxPool2d

        y5 = x.view(x.shape[0], -1)
        y5 = count_ones(y5)
        x = self.features[10](x)      #conv
        x = self.features[11](x)      #ReLU

        y6 = x.view(x.shape[0], -1)
        y6 = count_ones(y6)
        x = self.features[12](x)      #conv
        x = self.features[13](x)      #ReLU

        y7 = x.view(x.shape[0], -1)
        y7 = count_ones(y7)
        x = self.features[14](x)      #conv
        x = self.features[15](x)      #ReLU
        x = self.features[16](x)      #MaxPool2d

        y8 = x.view(x.shape[0], -1)
        y8 = count_ones(y8)
        x = self.features[17](x)      #conv
        x = self.features[18](x)      #ReLU

        y9 = x.view(x.shape[0], -1)
        y9 = count_ones(y9)
        x = self.features[19](x)      #conv
        x = self.features[20](x)      #ReLU

        y10 = x.view(x.shape[0], -1)
        y10 = count_ones(y10)
        x = self.features[21](x)      #conv
        x = self.features[22](x)      #ReLU
        x = self.features[23](x)      #MaxPool2d

        y11 = x.view(x.shape[0], -1)
        y11 = count_ones(y11)
        x = self.features[24](x)      #conv
        x = self.features[25](x)      #ReLU

        y12 = x.view(x.shape[0], -1)
        y12 = count_ones(y12)
        x = self.features[26](x)      #conv
        x = self.features[27](x)      #ReLU

        y13 = x.view(x.shape[0], -1)
        y13 = count_ones(y13)
        x = self.features[28](x)      #conv
        x = self.features[29](x)      #ReLU
        x = self.features[30](x)      #MaxPool2d

        x = x.view(x.size(0), -1)

        y14 = x.view(x.shape[0], -1)
        y14 = count_ones(y14)
        x = self.classifier[0](x)      #Linear
        x = self.classifier[1](x)      #ReLU
        x = self.classifier[2](x)      #Dropout

        y15 = x.view(x.shape[0], -1)
        y15 = count_ones(y15)
        x = self.classifier[3](x)      #Linear
        x = self.classifier[4](x)      #ReLU
        x = self.classifier[5](x)      #Dropout

        x = self.classifier[6](x)      #Linear

        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15

        return x, y
    
def count_ones(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return (activation_count - zeros)

def vgg16_stat(args=None):
    return VGG16_stat(args=args)
