import torch
import torch.nn as nn

class VGG16(nn.Module):
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
        
        # To compute sum of ones
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
        y1 = tanh(y1, self.beta)
        y1 = self.my_fc1(y1)
        x = self.features[0](x)
        x = self.features[1](x)

        y2 = x.view(x.shape[0], -1)
        y2 = tanh(y2, self.beta)
        y2 = self.my_fc2(y2)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)

        y3 = x.view(x.shape[0], -1)
        y3 = tanh(y3, self.beta)
        y3 = self.my_fc3(y3)
        x = self.features[5](x)
        x = self.features[6](x)

        y4 = x.view(x.shape[0], -1)
        y4 = tanh(y4, self.beta)
        y4 = self.my_fc4(y4)
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)

        y5 = x.view(x.shape[0], -1)
        y5 = tanh(y5, self.beta)
        y5 = self.my_fc5(y5)
        x = self.features[10](x)
        x = self.features[11](x)

        y6 = x.view(x.shape[0], -1)
        y6 = tanh(y6, self.beta)
        y6 = self.my_fc6(y6)
        x = self.features[12](x)
        x = self.features[13](x)

        y7 = x.view(x.shape[0], -1)
        y7 = tanh(y7, self.beta)
        y7 = self.my_fc7(y7)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)

        y8 = x.view(x.shape[0], -1)
        y8 = tanh(y8, self.beta)
        y8 = self.my_fc8(y8)
        x = self.features[17](x)
        x = self.features[18](x)

        y9 = x.view(x.shape[0], -1)
        y9 = tanh(y9, self.beta)
        y9 = self.my_fc9(y9)
        x = self.features[19](x)
        x = self.features[20](x)

        y10 = x.view(x.shape[0], -1)
        y10 = tanh(y10, self.beta)
        y10 = self.my_fc10(y10)
        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)

        y11 = x.view(x.shape[0], -1)
        y11 = tanh(y11, self.beta)
        y11 = self.my_fc11(y11)
        x = self.features[24](x)
        x = self.features[25](x)

        y12 = x.view(x.shape[0], -1)
        y12 = tanh(y12, self.beta)
        y12 = self.my_fc12(y12)
        x = self.features[26](x)
        x = self.features[27](x)

        y13 = x.view(x.shape[0], -1)
        y13 = tanh(y13, self.beta)
        y13 = self.my_fc13(y13)
        x = self.features[28](x)
        x = self.features[29](x)
        x = self.features[30](x)

        x = x.view(x.size(0), -1)

        y14 = x.view(x.shape[0], -1)
        y14 = tanh(y14, self.beta)
        y14 = self.my_fc14(y14)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)

        y15 = x.view(x.shape[0], -1)
        y15 = tanh(y15, self.beta)
        y15 = self.my_fc15(y15)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)

        x = self.classifier[6](x)

        y = self.my_fc16(torch.cat((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15),1))

        return x, y
    
    def vgg16_set_weights_one(self):

        self.my_fc1.weight.data.fill_(1.)
        self.my_fc2.weight.data.fill_(1.)
        self.my_fc3.weight.data.fill_(1.)
        self.my_fc4.weight.data.fill_(1.)
        self.my_fc5.weight.data.fill_(1.)
        self.my_fc6.weight.data.fill_(1.)
        self.my_fc7.weight.data.fill_(1.)
        self.my_fc8.weight.data.fill_(1.)
        self.my_fc9.weight.data.fill_(1.)
        self.my_fc10.weight.data.fill_(1.)
        self.my_fc11.weight.data.fill_(1.)
        self.my_fc12.weight.data.fill_(1.)
        self.my_fc13.weight.data.fill_(1.)
        self.my_fc14.weight.data.fill_(1.)
        self.my_fc15.weight.data.fill_(1.)
        self.my_fc16.weight.data.fill_(1.)

        return

def tanh(input_tensor, beta):
    output = torch.tanh(beta * input_tensor)
    output = torch.pow(output, 2)
    return output

def vgg16(args=None):
    return VGG16(args=args)