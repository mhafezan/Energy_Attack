import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1_stat(nn.Module):
    def __init__(self):
        super(Model1_stat, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        y1 = count_ones(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        y2 = count_ones(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        y3 = count_ones(x)
        x = self.conv3(x)
        x = F.relu(x)

        y4 = count_ones(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        y5 = count_ones(x)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        
        x = self.softmax(x)
        
        y = y1 + y2 + y3 + y4 + y5

        return x, y
  
def count_ones(input_tensor):
	return torch.count_nonzero(input_tensor).item()
    
