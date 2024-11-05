import torch
#from torchsummary import summary
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image 
import os
from torch.utils.data import DataLoader, random_split,Dataset
#import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import csv
import copy

import torch.nn as nn
import torchvision.utils as vutils

from optparse import OptionParser
import datetime

import sys
import gc

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]





def convert_to_batch_one():

    print('#################################################')
  
    #dataset = torch.load('../3_attack/adversarial_dataset_mnist_lr_0.5_with_l2_norm.pt', map_location=torch.device('cpu'))
    dataset = torch.load('../3_attack/adversarial_dataset_mnist_lr_0.7_no_l2_norm.pt', map_location=torch.device('cpu'))


    dataset_loader = DataLoader(
       dataset,
       batch_size=1,
       shuffle=False
       )
    
    print('dataset_loader=', len(dataset_loader))
  

    image_array = []


   
    
    #for batch_idx, (inputs, labels) in enumerate(dataset_loader):
    for index, (inputs, labels, adversarial) in enumerate(dataset_loader):       #enumerate(tqdm(test_dataset, desc='Data Progress')):

        #torch.cuda.memory_summary(device=None, abbreviated=False)
        #print(batch_idx)

        inputs = inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2), inputs.size(3), inputs.size(4))
        labels = labels.view(labels.size(0)*labels.size(1))
        
                            
        for ii in range(inputs.size(0)):
            tmpp = inputs[ii]
            #tmpp = tmpp.unsqueeze(0)

            image_array.append((tmpp, labels[ii], adversarial))





    image_array_dataset = AdversarialDataset(image_array)
    
    torch.save(image_array_dataset, './image_dataset_out.pt')
    




    return     


if __name__ == "__main__":
                 
    convert_to_batch_one()
    
 
     
    


            
