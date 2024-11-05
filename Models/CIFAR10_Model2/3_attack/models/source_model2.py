import torch
import torch.nn as nn
import torch.nn.functional as F

class Model2_stat(nn.Module):
    def __init__(self):
        super(Model2_stat, self).__init__()
        
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
        y1 = count_ones(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv layer
        y2 = count_ones(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv layer
        y3 = count_ones(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        # Fourth conv layer
        y4 = count_ones(x)
        x = self.conv4(x)
        x = F.relu(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # First fully connected layer
        y5 = count_ones(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        # Second fully connected layer
        x = self.fc2(x)
        
        y = y1 + y2 + y3 + y4 + y5
        
        return x, y
    
def count_ones(input_tensor):
	return torch.count_nonzero(input_tensor).item()
