import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import sys
import argparse
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

    
from models.source_c10_model import alexnet
from models.c10_model import alexnet_active

import datetime

import torchvision.datasets as dset
import torchvision.utils as vutils


# To compute Sparsity-Map for each layer's output
def copy_weights(model_source_in, model_dest_out):


    for name1, layer1 in model_source_in.named_modules():        

        if isinstance(layer1, nn.Conv2d) or isinstance(layer1, nn.BatchNorm2d) or isinstance(layer1, nn.ReLU) or isinstance(layer1, nn.MaxPool2d) or isinstance(layer1, nn.AdaptiveAvgPool2d) or isinstance(layer1, nn.Dropout2d) or isinstance(layer1, nn.Linear):
            
            for name2, layer2 in model_dest_out.named_modules():
                if name1 == name2:
                    #print(name1)
                    layer2.load_state_dict(layer1.state_dict())


    
    return


def test (net, c10_test_loader):

    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(c10_test_loader):

            image = image.cuda()
            label = label.cuda()

            #print(image)
            #print(torch.min(image))
            #print(torch.max(image))

            #img_grid = torchvision.utils.make_grid(image)
            #vutils.save_image(img_grid.detach(),
            #                  './sample_idx_%03d.png' % (n_iter),
            #                  normalize=True)
      
  
            
            output = net(image)
            _, pred = output[0].topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    #if args.gpu:
        #print('GPU INFO.....')
        #print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 accu: ",  correct_1 / len(c10_test_loader.dataset))
    #print("Top 5 accu: ", correct_5 / len(c10_test_loader.dataset))
    #print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


    return
    

if __name__ == '__main__':
       
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="AlexNet Network with CIFAR10 Dataset")
    parser.add_argument('--weights_source', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/cifar10/1_train_256_256/models/alexnet_cifar_50.pkl', help="The path to the saved weights")
    parser.add_argument('--weights_dest', default='./alexnet_cifar10_fc_one_out.pkl', help="The path to the saved weights")
    parser.add_argument('--batch_size', default=20, type=int, help="Batch size of 4 is a good choice for Training")
    parser.add_argument('--dataset', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/cifar10/cifar10_dataset', help="The path to the train and test datasets")

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_source = alexnet()
    model_dest = alexnet_active()

    model_source.to(device)
    model_dest.to(device)

    if torch.cuda.is_available():
        model_source.load_state_dict(torch.load(args.weights_source))
    else:
        model_source.load_state_dict(torch.load(args.weights_source, map_location=torch.device('cpu')))



    copy_weights(model_source, model_dest)

    model_dest.AlexNet_active_set_weights_one()

    torch.save(model_dest.state_dict(), f'./alexnet_cifar10_fc_one_out.pkl')    


    transform = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset  = dset.CIFAR10(root=args.dataset, train=False, transform=transform) # Original Testset

  


       
    test_loader = DataLoader(
       dataset,
       batch_size=args.batch_size,
       shuffle=False
       )



    ############################
   

    test(model_source, test_loader)

    test(model_dest, test_loader)
