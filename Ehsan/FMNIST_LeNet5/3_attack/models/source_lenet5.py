import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_stat(nn.Module):
    def __init__(self):
        super(LeNet5_stat, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        self.fc1 = nn.Linear(120, 84)
        
        self.fc2 = nn.Linear(84, 10)
        
        # To compute sum of ones
        self.my_fc1 = nn.Linear(1*32*32, 1, bias=False)
        self.my_fc2 = nn.Linear(6*14*14, 1, bias=False)
        self.my_fc3 = nn.Linear(16*5*5, 1, bias=False)
        self.my_fc4 = nn.Linear(120, 1, bias=False)
        self.my_fc5 = nn.Linear(4, 1, bias=False)
    
    def forward(self, x):
        
        # First convolution + activation + pooling
        y1 = x.view(x.shape[0], -1)
        y1 = count_ones(y1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second convolution + activation + pooling
        y2 = x.view(x.shape[0], -1)
        y2 = count_ones(y2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third convolution + activation
        y3 = x.view(x.shape[0], -1)
        y3 = count_ones(y3)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(-1, 120)
        
        y4 = x.view(x.shape[0], -1)
        y4 = count_ones(y4)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        y = y1 + y2 + y3 + y4
        
        return x, y
    
def count_ones(input_tensor):
    return torch.count_nonzero(input_tensor)

def lenet5_stat():
    return LeNet5_stat()
    
