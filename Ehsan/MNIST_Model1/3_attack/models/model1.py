import torch
import torch.nn as nn

class LeNet5_active(nn.Module):
    def __init__(self, beta=15):
        super(LeNet5_active, self).__init__()
        self.beta  = beta
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3   = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

        # Each my_fcX is designed based on the size of the tensor introduced to each convolution or linear layer
        self.my_fc1 = nn.Linear(1*28*28, 1, bias=False)
        self.my_fc2 = nn.Linear(6*12*12, 1, bias=False)
        self.my_fc3 = nn.Linear(256, 1, bias=False)
        self.my_fc4 = nn.Linear(120, 1, bias=False)
        self.my_fc5 = nn.Linear(84, 1, bias=False)

        self.my_fc6 = nn.Linear(5, 1, bias=False)
        
    def forward(self, x):
        
        y1 = x.view(x.shape[0], -1)
        y1 = tanh(y1, self.beta)
        y1 = self.my_fc1(y1)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        y2 = x.view(x.shape[0], -1)
        y2 = tanh(y2, self.beta)
        y2 = self.my_fc2(y2)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = x.view(x.shape[0], -1)

        y3 = x.view(x.shape[0], -1)
        y3 = tanh(y3, self.beta)        
        y3 = self.my_fc3(y3)
        
        x = self.fc1(x)
        x = self.relu3(x)

        y4 = x.view(x.shape[0], -1)
        y4 = tanh(y4, self.beta)
        y4 = self.my_fc4(y4)
        
        x = self.fc2(x)
        x = self.relu4(x)

        y5 = x.view(x.shape[0], -1)
        y5 = tanh(y5, self.beta)
        y5 = self.my_fc5(y5)
        
        x = self.fc3(x)
        x = self.relu5(x)

        y = self.my_fc6(torch.cat((y1,y2,y3,y4,y5), 1)) # Y elements represents the number of non-zeros in a batch
        
        return x, y

    def LeNet5_active_set_weights_one(self):

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

def lenet5_active(beta=15):
    return LeNet5_active(beta=beta)