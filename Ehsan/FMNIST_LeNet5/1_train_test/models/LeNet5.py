"""
Customized LeNet5 gotten from the paper: Classification of Garments from Fashion MNIST Dataset Using CNN LeNet-5 Architecture
"""

import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # First convolutional layer (C1): Input 32x32x1, Output 28x28x6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        # Second layer (S2): Average Pooling Layer, Input 28x28x6, Output 14x14x6
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Third convolutional layer (C3): Input 14x14x6, Output 10x10x16
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        # Fourth layer (S4): Average Pooling Layer, Input 10x10x16, Output 5x5x16
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fifth layer (C5): Fully connected convolutional layer, Input 5x5x16, Output 1x1x120
        # Since the input size is reduced to 5x5, we use kernel size 5 to connect it to C5
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        # Sixth layer (F6): Fully connected layer, Input 120, Output 84
        self.fc1 = nn.Linear(120, 84)
        
        # Output layer: Fully connected layer, Input 84, Output 10
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self, x):
        
        # First convolution + activation + pooling
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.tanh(x)
        x = self.pool1(x)
        
        # Second convolution + activation + pooling
        x = self.conv2(x)
        x = F.relu(x)
        # x = F.tanh(x)
        x = self.pool2(x)
        
        # Third convolution + activation
        x = self.conv3(x)
        x = F.relu(x)
        # x = F.tanh(x)
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 120)
        
        # Fully connected layers with activation
        x = self.fc1(x)
        x = F.relu(x)
        # x = F.tanh(x)
        
        # Output layer (Softmax is applied through Cross-Entropy loss function)
        x = self.fc2(x)
        
        return x
    