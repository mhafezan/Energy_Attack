import torch
import torch.nn as nn
import torch.nn.functional as F

class Model3_stat(nn.Module):
    def __init__(self, num_classes=100):
        super(Model3_stat, self).__init__()
        
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
        
        y1 = count_ones(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        y2 = count_ones(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        y3 = count_ones(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        
        y4 = count_ones(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        y = y1 + y2 + y3 + y4
        
        return x, y
    
def count_ones(input_tensor):
	return torch.count_nonzero(input_tensor).item()
