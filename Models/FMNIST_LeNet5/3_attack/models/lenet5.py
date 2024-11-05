import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10, args=None):
        super(LeNet5, self).__init__()
        
        self.beta = args.beta
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        
        self.fc1 = nn.Linear(120, 84)
        
        self.fc2 = nn.Linear(84, num_classes)
        
        # To compute sum of ones
        self.my_fc1 = nn.Linear(1*32*32, 1, bias=False)
        self.my_fc2 = nn.Linear(6*14*14, 1, bias=False)
        self.my_fc3 = nn.Linear(16*5*5, 1, bias=False)
        self.my_fc4 = nn.Linear(120, 1, bias=False)
        self.my_fc5 = nn.Linear(4, 1, bias=False)
    
    def forward(self, x):
        
        # First convolution + activation + pooling
        y1 = x.view(x.shape[0], -1)
        y1 = tanh(y1, self.beta)
        y1 = self.my_fc1(y1)
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second convolution + activation + pooling
        y2 = x.view(x.shape[0], -1)
        y2 = tanh(y2, self.beta)
        y2 = self.my_fc2(y2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third convolution + activation
        y3 = x.view(x.shape[0], -1)
        y3 = tanh(y3, self.beta)
        y3 = self.my_fc3(y3)
        
        x = self.conv3(x)
        x = F.relu(x)
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 120)
        
        # Fully connected layers with activation
        y4 = x.view(x.shape[0], -1)
        y4 = tanh(y4, self.beta)
        y4 = self.my_fc4(y4)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        # Output layer (Softmax is applied through Cross-Entropy loss function)
        x = self.fc2(x)
        
        y = self.my_fc5(torch.cat((y1, y2, y3, y4), 1))
        
        return x, y

def tanh(input_tensor, beta):
    output = torch.tanh(beta * input_tensor)
    output = torch.pow(output, 2)
    return output

def lenet5(args=None):
    return LeNet5(args=args)
    