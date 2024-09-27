import torch
import torch.nn as nn


 

class AlexNet_active(nn.Module):
    def __init__(self, beta=15):
        super(AlexNet_active, self).__init__()

        self.beta  = beta

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                 
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )

        #458624
        self.my_fc1 = nn.Linear(3*256*256, 1, bias=False)
        self.my_fc2 = nn.Linear(64*31*31, 1, bias=False)
        self.my_fc3 = nn.Linear(192*15*15, 1, bias=False)
        self.my_fc4 = nn.Linear(384*15*15, 1, bias=False)
        self.my_fc5 = nn.Linear(256*15*15, 1, bias=False)

        self.my_fc6 = nn.Linear(9216, 1, bias=False)
        self.my_fc7 = nn.Linear(4096, 1, bias=False)

        self.my_fc8 = nn.Linear(7, 1, bias=False)
  
        
      
    def forward(self, x):
       
        
        #print(x.size())
        y1 = x.view(x.shape[0], -1)
        y1 = tanh(y1, self.beta)
        y1 = self.my_fc1(y1)

        x = self.features[0](x)  #conv
        x = self.features[1](x)  #relu        
        x = self.features[2](x)  #maxpool
      

       #print(x.size())
        y2 = x.view(x.shape[0], -1)
        y2 = tanh(y2, self.beta)
        y2 = self.my_fc2(y2)
 
        x = self.features[3](x)  #conv
        x = self.features[4](x)  #relu        
        x = self.features[5](x)  #maxpool
       
        #print(x.size())
        y3 = x.view(x.shape[0], -1)
        y3 = tanh(y3, self.beta)
        y3 = self.my_fc3(y3)
 
        x = self.features[6](x)  #conv
        x = self.features[7](x)  #relu
        
      
        #print(x.size())
        y4 = x.view(x.shape[0], -1)
        y4 = tanh(y4, self.beta)
        y4 = self.my_fc4(y4)

        x = self.features[8](x)  #conv
        x = self.features[9](x)  #relu
        

        #print(x.size())
        y5 = x.view(x.shape[0], -1)
        y5 = tanh(y5, self.beta)
        y5 = self.my_fc5(y5)

        x = self.features[10](x) #conv
        x = self.features[11](x) #relu        
        x = self.features[12](x) #maxpool     
        
        x = self.avgpool(x)

        x = x.view(x.size(0), 256 * 6 * 6)    

        x = self.classifier[0](x) #drop

        #print(x.size())
        y6 = x.view(x.shape[0], -1)
        y6 = tanh(y6, self.beta)
        y6 = self.my_fc6(y6)

        x = self.classifier[1](x) #linear           
        x = self.classifier[2](x) #relu
        
        x = self.classifier[3](x) #drop


        #print(x.size())
        y7 = x.view(x.shape[0], -1)
        y7 = tanh(y7, self.beta)
        y7 = self.my_fc7(y7)
 
        
        x = self.classifier[4](x) #linear
        x = self.classifier[5](x) #relu
        
        x = self.classifier[6](x) #linear


        y = self.my_fc8( torch.cat((y1,y2,y3,y4,y5,y6,y7),1) )

        
        return x, y

  

    def AlexNet_active_set_weights_one(self):

        self.my_fc1.weight.data.fill_(1.)
        self.my_fc2.weight.data.fill_(1.)
        self.my_fc3.weight.data.fill_(1.)
        self.my_fc4.weight.data.fill_(1.)
        self.my_fc5.weight.data.fill_(1.)
        self.my_fc6.weight.data.fill_(1.)
        self.my_fc7.weight.data.fill_(1.)
        self.my_fc8.weight.data.fill_(1.)

        return
  


def tanh(input_tensor, beta):
    # To scale the tensor by BETA and apply tanh function to the scaled tensor
    output = torch.tanh(beta * input_tensor)
    # To sum the activations separately for each image in the batch
    return output #output.view(input_tensor.size(0), -1).sum(dim=1)


def alexnet_active(beta=15):
    return AlexNet_active(beta=beta)
