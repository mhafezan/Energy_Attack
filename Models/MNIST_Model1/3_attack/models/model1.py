import torch
import torch.nn as nn
from torch.nn import functional as F

class Model1(nn.Module):
    def __init__(self, args=None):
        super(Model1, self).__init__()
        
        self.beta = args.beta

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax(dim=1)
        
        # To compute sum of ones
        self.my_fc1 = nn.Linear(1*28*28, 1, bias=False)
        self.my_fc2 = nn.Linear(32*14*14, 1, bias=False)
        self.my_fc3 = nn.Linear(32*7*7, 1, bias=False)
        self.my_fc4 = nn.Linear(64*7*7, 1, bias=False)
        self.my_fc5 = nn.Linear(3136, 1, bias=False)
        self.my_fc6 = nn.Linear(5, 1, bias=False)

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
        x = self.pool(x)

        y3 = x.view(x.shape[0], -1)
        y3 = tanh(y3, self.beta)
        y3 = self.my_fc3(y3)
        x = self.conv3(x)
        x = F.relu(x)

        y4 = x.view(x.shape[0], -1)
        y4 = tanh(y4, self.beta)
        y4 = self.my_fc4(y4)
        x = self.conv4(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1)

        y5 = x.view(x.shape[0], -1)
        y5 = tanh(y5, self.beta)
        y5 = self.my_fc5(y5)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        
        x = self.softmax(x)
        
        y = self.my_fc6(torch.cat((y1,y2,y3,y4,y5), 1))

        return x, y

    def model1_active_set_weights_one(self):
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
    
