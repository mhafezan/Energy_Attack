import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj, fc_in):
        super().__init__()
        
        # 1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1conv -> 5x5conv branch
        # We use two 3x3 conv filters stacked instead of 1 5x5 filters to obtain the same receptive field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        # 3x3 pooling -> 1x1conv
        # With the same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

        # for computing sume of ones
        self.my_fc = nn.ModuleList([
            nn.Linear(input_channels*fc_in*fc_in, 1, bias=False),
            nn.Linear(n1x1*fc_in*fc_in, 1, bias=False),
            nn.Linear(n3x3_reduce*fc_in*fc_in, 1, bias=False),
            nn.Linear(n3x3*fc_in*fc_in, 1, bias=False),
            nn.Linear(n5x5_reduce*fc_in*fc_in, 1, bias=False),
            nn.Linear(n5x5*fc_in*fc_in, 1, bias=False),
            nn.Linear(input_channels*fc_in*fc_in, 1, bias=False),
            
            nn.Linear(7, 1, bias=False) 
        ])
      
    def forward(self, x):

        zeros    = []
        sizes    = []
        smaps    = []

        b1_zeros, b1_sizes = sparsity_rate(x)
        zeros.append(b1_zeros)
        sizes.append(b1_sizes)
        smaps.append(sparsity_map(x))
        s = self.b1[0](x) # conv2
        s = self.b1[1](s) # batch
        s = self.b1[2](s) # relu

        # We don't need to compute SR(x) and SM(x) at this point, as we already computed it at b1 layer
        t = self.b2[0](x) # conv2
        t = self.b2[1](t) # batch
        t = self.b2[2](t) # relu

        b2_zeros, b2_sizes = sparsity_rate(t)
        zeros.append(b2_zeros)
        sizes.append(b2_sizes)
        smaps.append(sparsity_map(t))
        t = self.b2[3](t) # conv2
        t = self.b2[4](t) # batch
        t = self.b2[5](t) # relu

        # We don't need to compute SR(x) and SM(x) at this point, as we already computed it at b1 layer
        u = self.b3[0](x) # conv2
        u = self.b3[1](u) # batch
        u = self.b3[2](u) # relu

        b3_zeros, b3_sizes = sparsity_rate(u)
        zeros.append(b3_zeros)
        sizes.append(b3_sizes)
        smaps.append(sparsity_map(u))
        u = self.b3[3](u) # conv2
        u = self.b3[4](u) # batch
        u = self.b3[5](u) # relu

        b3_zeros, b3_sizes = sparsity_rate(u)
        zeros.append(b3_zeros)
        sizes.append(b3_sizes)
        smaps.append(sparsity_map(u))
        u = self.b3[6](u) # conv2
        u = self.b3[7](u) # batch
        u = self.b3[8](u) # relu
     
        v = self.b4[0](x) # MaxPool2d

        b4_zeros, b4_sizes = sparsity_rate(v)
        zeros.append(b4_zeros)
        sizes.append(b4_sizes)
        smaps.append(sparsity_map(v))
        v = self.b4[1](v) # conv2
        v = self.b4[2](v) # batch
        v = self.b4[3](v) # relu

        return torch.cat([s, t, u, v], dim=1), zeros, sizes, smaps
    
class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True))

        # Although we only use 1 conv layer as prelayer, we still use name a3, b3, ...
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, 16)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, 16)

        # In general, an Inception network is a network consisting of modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the grid
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, 8)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64, 8)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, 8)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, 8)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, 8)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, 4)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, 4)

        # Input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)
       
        # For computing sum of ones
        self.my_fc_pl_0 = nn.Linear(3*32*32,  1, bias=False)
        self.my_fc_pl_1 = nn.Linear(64*32*32, 1, bias=False)
        self.my_fc_pl_2 = nn.Linear(64*32*32, 1, bias=False)

        self.my_fc_12 = nn.Linear(12, 1, bias=False)

    def forward(self, x):
        # We break forward function into the minor layers: refer to https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/googlenet.py for the original function
        
        # To initialize the zeros and sizes with zero value
        L1_zeros, L2_zeros, L3_zeros, L49_zeros = [0] * 4
        L4_L8_zeros, L9_L13_zeros, L14_L18_zeros, L19_L23_zeros, L24_L28_zeros, L29_L33_zeros, L34_L38_zeros, L39_L43_zeros, L44_L48_zeros = ([] for _ in range(9))
        L1_sizes, L2_sizes, L3_sizes, L49_sizes = [0] * 4
        L4_L8_sizes, L9_L13_sizes, L14_L18_sizes, L19_L23_sizes, L24_L28_sizes, L29_L33_sizes, L34_L38_sizes, L39_L43_sizes, L44_L48_sizes = ([] for _ in range(9))
        
        # We don't compute SR for the first convolution input, as SR is 0% for colored images
        L1_zeros, L1_sizes = [0] * 2
        L1_smap = sparsity_map(x)
        x = self.prelayer[0](x) # conv2
        x = self.prelayer[1](x) # batch
        x = self.prelayer[2](x) # relu

        L2_zeros, L2_sizes = sparsity_rate(x)
        L2_smap = sparsity_map(x)
        x = self.prelayer[3](x) # conv2
        x = self.prelayer[4](x) # batch
        x = self.prelayer[5](x) # relu

        L3_zeros, L3_sizes = sparsity_rate(x)
        L3_smap = sparsity_map(x)
        x = self.prelayer[6](x) # conv2
        x = self.prelayer[7](x) # batch
        x = self.prelayer[8](x) # relu

        x = self.maxpool(x)

        x, L4_L8_zeros, L4_L8_sizes, L4_L8_smap = self.a3(x)
        
        x, L9_L13_zeros, L9_L13_sizes, L9_L13_smap = self.b3(x)

        x = self.maxpool(x)

        x, L14_L18_zeros, L14_L18_sizes, L14_L18_smap = self.a4(x)

        x, L19_L23_zeros, L19_L23_sizes, L19_L23_smap = self.b4(x)

        x, L24_L28_zeros, L24_L28_sizes, L24_L28_smap = self.c4(x)
        
        x, L29_L33_zeros, L29_L33_sizes, L29_L33_smap = self.d4(x)

        x, L34_L38_zeros, L34_L38_sizes, L34_L38_smap = self.e4(x)

        x = self.maxpool(x)

        x, L39_L43_zeros, L39_L43_sizes, L39_L43_smap = self.a5(x)

        x, L44_L48_zeros, L44_L48_sizes, L44_L48_smap = self.b5(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        
        # Concatenate the gathered flattened tensors to generate a unified sparsity-map for entire inference
        L1_smap_flat = L1_smap.view(-1)
        L2_smap_flat = L2_smap.view(-1)
        L3_smap_flat = L3_smap.view(-1)
        L4_L8_smap_flat   = [tensor.view(-1) for tensor in L4_L8_smap]
        L9_L13_smap_flat  = [tensor.view(-1) for tensor in L9_L13_smap]
        L14_L18_smap_flat = [tensor.view(-1) for tensor in L14_L18_smap]
        L19_L23_smap_flat = [tensor.view(-1) for tensor in L19_L23_smap]
        L24_L28_smap_flat = [tensor.view(-1) for tensor in L24_L28_smap]
        L29_L33_smap_flat = [tensor.view(-1) for tensor in L29_L33_smap]
        L34_L38_smap_flat = [tensor.view(-1) for tensor in L34_L38_smap]
        L39_L43_smap_flat = [tensor.view(-1) for tensor in L39_L43_smap]
        L44_L48_smap_flat = [tensor.view(-1) for tensor in L44_L48_smap]
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat] + L4_L8_smap_flat + L9_L13_smap_flat + L14_L18_smap_flat + L19_L23_smap_flat
                                        + L24_L28_smap_flat + L29_L33_smap_flat + L34_L38_smap_flat + L39_L43_smap_flat + L44_L48_smap_flat, dim=0)
        
        zeros_list = [L1_zeros] + [L2_zeros] + [L3_zeros] + L4_L8_zeros + L9_L13_zeros + L14_L18_zeros + L19_L23_zeros + L24_L28_zeros + L29_L33_zeros + L34_L38_zeros + L39_L43_zeros + L44_L48_zeros
        sizes_list = [L1_sizes] + [L2_sizes] + [L3_sizes] + L4_L8_sizes + L9_L13_sizes + L14_L18_sizes + L19_L23_sizes + L24_L28_sizes + L29_L33_sizes + L34_L38_sizes + L39_L43_sizes + L44_L48_sizes
        
        return x, zeros_list, sizes_list, sparsity_map_total

def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

# To compute Sparsity-Map for each layer's output
def sparsity_map(input_tensor):
    # To create a tensor with all zeros and with the same shape of input tensor
    sparsity_map = torch.zeros_like(input_tensor, dtype=torch.uint8)
    # To find the positions where the input tensor is zero
    zero_positions = torch.eq(input_tensor, 0)
    # To set the corresponding positions in the bit_map to 1
    sparsity_map = sparsity_map.masked_fill(zero_positions, 1)
    return sparsity_map

def calculate_range(current_sparsity_list, sparsity_range, class_index):
    
    for layer in range (len(current_sparsity_list)):

        if current_sparsity_list[layer] < sparsity_range[class_index][layer][0]:
            sparsity_range[class_index][layer][0] = current_sparsity_list[layer]

        if current_sparsity_list[layer] > sparsity_range[class_index][layer][1]:
            sparsity_range[class_index][layer][1] = current_sparsity_list[layer]

    return sparsity_range

def Profile (model, device, train_loader, num_classes, args):
    
    updated_maps  = [None] * num_classes
    diff_num_ones = [0] * num_classes
    prev_num_ones = [0] * num_classes
    curr_num_ones = [0] * num_classes
    num_layers = 48
    
    # To define sparsity-range for each class
    range_for_each_class = [[[float('inf'), float('-inf')] for _ in range(num_layers)] for _ in range(num_classes)]

    for index, (data, target) in enumerate(tqdm(train_loader, desc='Data Progress')):
        
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)

        pred = output[0].max(1, keepdim=False)[1]
        if pred.item() == target.item():
            if args.method == 'sparsity-range':
                current_sparsity_rate = [0.0] * num_layers  # Initialize the current_sparsity list reading each model input
                for i in range(num_layers):
                    current_sparsity_rate[i] = (output[1][i] / output[2][i]) if output[2][i] != 0 else 0
                range_for_each_class = calculate_range(current_sparsity_rate, range_for_each_class, target.item())

            elif args.method == 'sparsity-map':
                if updated_maps[target.item()] == None:
                    updated_maps[target.item()] = output[3]
                    prev_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item() # Computes the number of 1s in the updated_maps
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                else:
                    updated_maps [target.item()] = torch.bitwise_or(updated_maps[target.item()], output[3]) # output[3] is the Sparsity-Map associated with the current input
                    curr_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()]) # Computes the difference between the number of 1s in the current and previous Sparsity-Maps
                    prev_num_ones[target.item()] = curr_num_ones[target.item()]

                if all(num < 1 for num in diff_num_ones):
                    print(diff_num_ones)
                    break

    return updated_maps, range_for_each_class, index

if __name__ == '__main__':

    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="GoogleNet Network with CIFAR100 Dataset")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism: sparsity-map or sparsity-range")
    parser.add_argument('--weights', default="../2_copy_weight/googlenet_cifar100_fc_one_out.pkl", help="The path to the pre-trained weights")
    parser.add_argument('--dataset', default="../cifar100_dataset", help="The path to the train or test datasets")
    args = parser.parse_args()
    print(f"{args}\n")
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n {device} is assigned for processing! \n")

    # CIFAR100 dataset and dataloader declaration
    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    train_dataset = torchvision.datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        
    # Model Initialization
    model = GoogleNet().to(device)

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()
        
    # Set the model in evaluation mode
    model.eval()
        
    num_classes  = len(train_dataset.classes)

    # The profile function returns an array in which each element comprises a Sparsity-Map Tensor assigned to each Class
    sparsity_maps, sparsity_range, index = Profile (model, device, train_loader, num_classes, args)
        
    # Store the Sparsity_Maps in an offline file
    print(f"We should stop profiling at data-point {index} when profiling is based on {args.method}")
    print(f"P1 = {((index+1)/len(train_loader))*100:.2f} % of training-set has been used for profiling.")
    
    if not os.path.isdir('profile_data'):
        os.mkdir('profile_data')

    if args.method == 'sparsity-map':
        # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
        torch.save(sparsity_maps, "./profile_data/sparsity_maps.pt", pickle_protocol=4)
        
    if args.method == 'sparsity-range':
        torch.save(sparsity_range, "./profile_data/sparsity_ranges.pt", pickle_protocol=4)

    sys.exit(0)

# Profile: python3 cifar100_profile.py --method sparsity-map/range --dataset ../cifar100_dataset --weights ../2_copy_weight/googlenet_cifar100_fc_one_out.pkl