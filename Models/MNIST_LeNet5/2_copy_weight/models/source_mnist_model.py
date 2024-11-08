import torch
import torch.nn as nn



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
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
       
      
        
    def forward(self, x):
                
        #print(x.size())
        y1 = count_ones(x)  
       
        x = self.conv1(x)
        x = self.relu1(x)    
        x = self.pool1(x)


        #print(x.size())
        y2 = count_ones(x)  
        
        x = self.conv2(x)        
        x = self.relu2(x)      
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)

        
        #print(x.size())
        y3 = count_ones(x)  
      
        x = self.fc1(x)
        x = self.relu3(x)

       
        #print(x.size())
        y4 = count_ones(x)  

        x = self.fc2(x)
        x = self.relu4(x)

      
        #print(x.size())
        y5 = count_ones(x)  
       
        x = self.fc3(x)
        x = self.relu5(x)

        y = y1 + y2 + y3 + y4 + y5      

        return x, y

  
def count_ones(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return (activation_count - zeros)


def lenet5():
    return LeNet5()
