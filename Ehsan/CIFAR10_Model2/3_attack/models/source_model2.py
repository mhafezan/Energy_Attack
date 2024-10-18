import torch
import torch.nn as nn

class Model2_stat(nn.Module):
    def __init__(self, args=None):
        super(Model2_stat, self).__init__()

        self.beta  = args.beta

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
        y1 = count_ones(x)  
      
        x = self.features[0](x)  #conv
        x = self.features[1](x)  #relu        
        x = self.features[2](x)  #maxpool
      

        #print(x.size())
        y2 = count_ones(x)  

        x = self.features[3](x)  #conv
        x = self.features[4](x)  #relu        
        x = self.features[5](x)  #maxpool
       
        #print(x.size())
        y3 = count_ones(x)  

        x = self.features[6](x)  #conv
        x = self.features[7](x)  #relu
        
      
        #print(x.size())
        y4 = count_ones(x)  


        x = self.features[8](x)  #conv
        x = self.features[9](x)  #relu
        

        #print(x.size())
        y5 = count_ones(x)  


        x = self.features[10](x) #conv
        x = self.features[11](x) #relu        
        x = self.features[12](x) #maxpool     
        
        x = self.avgpool(x)      

        x = x.view(x.size(0), 256 * 6 * 6)    

        
        x = self.classifier[0](x) #drop

        #print(x.size())
        y6 = count_ones(x)  


        x = self.classifier[1](x) #linear           
        x = self.classifier[2](x) #relu
        
        x = self.classifier[3](x) #drop


        #print(x.size())
        y7 = count_ones(x)  

 
        
        x = self.classifier[4](x) #linear
        x = self.classifier[5](x) #relu
        
        x = self.classifier[6](x) #linear


        y = y1+y2+y3+y4+y5+y6+y7

        
        return x, y
 
def count_ones(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return (activation_count - zeros)

