import torch
import torch.nn as nn
from torch.nn import functional as F

class Model2(nn.Module):
    def __init__(self, args=None):
        super(Model2, self).__init__()
        
        self.beta = args.beta
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        self.fc1 = nn.Linear(32*22*22, 512)
        self.fc2 = nn.Linear(512, 10)
        
        # To compute sum of ones
        self.my_fc1 = nn.Linear(3*256*256, 1, bias=False)
        self.my_fc2 = nn.Linear(64*84*84, 1, bias=False)
        self.my_fc3 = nn.Linear(64*26*26, 1, bias=False)
        self.my_fc4 = nn.Linear(64*24*24, 1, bias=False)
        self.my_fc5 = nn.Linear(32*22*22, 1, bias=False)
        self.my_fc6 = nn.Linear(5, 1, bias=False)
    
    def forward(self, x):
        
        # First conv layer
        y1 = x.view(x.shape[0], -1)
        y1 = tanh(y1, self.beta)
        y1 = self.my_fc1(y1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv layer
        y2 = x.view(x.shape[0], -1)
        y2 = tanh(y2, self.beta)
        y2 = self.my_fc2(y2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv layer
        y3 = x.view(x.shape[0], -1)
        y3 = tanh(y3, self.beta)
        y3 = self.my_fc3(y3)
        x = self.conv3(x)
        x = F.relu(x)
        
        # Fourth conv layer
        y4 = x.view(x.shape[0], -1)
        y4 = tanh(y4, self.beta)
        y4 = self.my_fc4(y4)
        x = self.conv4(x)
        x = F.relu(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # First fully connected layer
        y5 = x.view(x.shape[0], -1)
        y5 = tanh(y5, self.beta)
        y5 = self.my_fc5(y5)
        x = self.fc1(x)
        x = F.relu(x)
        
        # Second fully connected layer
        x = self.fc2(x)
        
        y = self.my_fc6(torch.cat((y1,y2,y3,y4,y5), 1))
        
        return x, y
    
    def model2_set_weights_one(self):
        self.my_fc1.weight.data.fill_(1.)
        self.my_fc2.weight.data.fill_(1.)
        self.my_fc3.weight.data.fill_(1.)
        self.my_fc4.weight.data.fill_(1.)
        self.my_fc5.weight.data.fill_(1.)
        self.my_fc6.weight.data.fill_(1.)
        return

def tanh(input_tensor, beta):
    output = torch.tanh(beta * input_tensor)
    output = torch.pow(output, 2)
    return output
