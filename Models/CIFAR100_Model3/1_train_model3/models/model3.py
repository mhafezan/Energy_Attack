import torch.nn as nn
import torch.nn.functional as F

class Model3(nn.Module):
    def __init__(self, num_classes=100):
        super(Model3, self).__init__()
        
        # To define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        
        # To Define the fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 64)  # CIFAR-100 images are 32x32; downsampled by pooling
        self.fc2 = nn.Linear(64, num_classes)
        
        # To define max pooling layer and softmax
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return x
        
