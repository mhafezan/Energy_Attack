import torch
import torch.nn as nn
from torch.nn import functional as F

class Model3(nn.Module):
    def __init__(self, num_classes=100, args=None):
        super(Model3, self).__init__()
        
        self.beta = args.beta
        
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # To compute sum of ones
        self.my_fc1 = nn.Linear(3*32*32, 1, bias=False)
        self.my_fc2 = nn.Linear(32*16*16, 1, bias=False)
        self.my_fc3 = nn.Linear(32*16*16, 1, bias=False)
        self.my_fc4 = nn.Linear(8192, 1, bias=False)
        self.my_fc5 = nn.Linear(4, 1, bias=False)
    
    def forward(self, x):
        
        y1 = x.view(x.shape[0], -1)
        y1 = tanh(y1, self.beta)
        y1 = self.my_fc1(y1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        y2 = x.view(x.shape[0], -1)
        y2 = tanh(y2, self.beta)
        y2 = self.my_fc2(y2)
        x = self.conv2(x)
        x = F.relu(x)
        
        y3 = x.view(x.shape[0], -1)
        y3 = tanh(y3, self.beta)
        y3 = self.my_fc3(y3)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        
        y4 = x.view(x.shape[0], -1)
        y4 = tanh(y4, self.beta)
        y4 = self.my_fc4(y4)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        y = self.my_fc5(torch.cat((y1,y2,y3,y4), 1))
        
        return x, y
        
    def model3_set_weights_one(self):
        self.my_fc1.weight.data.fill_(1.)
        self.my_fc2.weight.data.fill_(1.)
        self.my_fc3.weight.data.fill_(1.)
        self.my_fc4.weight.data.fill_(1.)
        self.my_fc5.weight.data.fill_(1.)
        return
        
def tanh(input_tensor, beta):
    output = torch.tanh(beta * input_tensor)
    output = torch.pow(output, 2)
    return output
