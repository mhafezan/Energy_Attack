import argparse
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Custom Dataset to store the adversarial data
class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj, beta=15):
        super().__init__()
        self.beta = beta

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

    def forward(self, x):
        self.zeros = []
        self.sizes = []
        self.tanhs = []
        self.smaps = [] # To gather an array of tensors
        
        s = self.b1(x)
        zz, ss = sparsity_rate(s)
        self.tanhs.append(tanh(s, self.beta))
        self.smaps.append(sparsity_map(s))
        self.zeros.append(zz)
        self.sizes.append(ss)
        
        t = self.b2[0](x)
        t = self.b2[1](t)
        t = self.b2[2](t)
        zz, ss = sparsity_rate(t)
        self.tanhs.append(tanh(t, self.beta))
        self.smaps.append(sparsity_map(t))
        self.zeros.append(zz)
        self.sizes.append(ss)
        t = self.b2[3](t)
        t = self.b2[4](t)
        t = self.b2[5](t)
        zz, ss = sparsity_rate(t)
        self.tanhs.append(tanh(t, self.beta))
        self.smaps.append(sparsity_map(t))
        self.zeros.append(zz)
        self.sizes.append(ss)
        
        u = self.b3[0](x)
        u = self.b3[1](u)
        u = self.b3[2](u)
        zz, ss = sparsity_rate(u)
        self.tanhs.append(tanh(u, self.beta))
        self.smaps.append(sparsity_map(u))
        self.zeros.append(zz)
        self.sizes.append(ss)
        u = self.b3[3](u)
        u = self.b3[4](u)
        u = self.b3[5](u)
        zz, ss = sparsity_rate(u)
        self.tanhs.append(tanh(u, self.beta))
        self.smaps.append(sparsity_map(u))
        self.zeros.append(zz)
        self.sizes.append(ss)
        u = self.b3[6](u)
        u = self.b3[7](u)
        u = self.b3[8](u)
        zz, ss = sparsity_rate(u)
        self.tanhs.append(tanh(u, self.beta))
        self.smaps.append(sparsity_map(u))
        self.zeros.append(zz)
        self.sizes.append(ss)
        
        v = self.b4(x)
        zz, ss = sparsity_rate(v)
        self.tanhs.append(tanh(v, self.beta))
        self.smaps.append(sparsity_map(v))
        self.zeros.append(zz)
        self.sizes.append(ss)
        
        return torch.cat([s, t, u, v], dim=1), self.zeros, self.sizes, self.tanhs, self.smaps

class GoogleNet(nn.Module):

    def __init__(self, num_class=100, beta=15):
        super().__init__()
        self.beta = beta
        
        self.prelayer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        # Although we only use 1 conv layer as prelayer, we still use name a3, b3, ...
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, self.beta)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, self.beta)

        # In general, an Inception network is a network consisting of modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the grid
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, self.beta)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64, self.beta)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, self.beta)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, self.beta)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, self.beta)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, self.beta)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, self.beta)

        # Input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)
    
    # We break forward function into the minor layers: refer to https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/googlenet.py for the original code
    def forward(self, x):
        
        x = self.prelayer[0](x)
        x = self.prelayer[1](x)
        x = self.prelayer[2](x)
        L1_zeros, L1_sizes = sparsity_rate(x)
        L1_tanh = tanh(x, self.beta)
        L1_smap = sparsity_map(x)
        x = self.prelayer[3](x)
        x = self.prelayer[4](x)
        x = self.prelayer[5](x)
        L2_zeros, L2_sizes = sparsity_rate(x)
        L2_tanh = tanh(x, self.beta)
        L2_smap = sparsity_map(x)
        x = self.prelayer[6](x)
        x = self.prelayer[7](x)
        x = self.prelayer[8](x)
        L3_zeros, L3_sizes = sparsity_rate(x)
        L3_tanh = tanh(x, self.beta)
        L3_smap = sparsity_map(x)

        x = self.maxpool(x)
        L4_zeros, L4_sizes = sparsity_rate(x)
        L4_tanh = tanh(x, self.beta)
        L4_smap = sparsity_map(x)

        x, L5_L11_zeros, L5_L11_sizes, L5_L11_tanh, L5_L11_smap = self.a3(x)

        x, L12_L18_zeros, L12_L18_sizes, L12_L18_tanh, L12_L18_smap = self.b3(x)

        x = self.maxpool(x)
        L19_zeros, L19_sizes = sparsity_rate(x)
        L19_tanh = tanh(x, self.beta)
        L19_smap = sparsity_map(x)

        x, L20_L26_zeros, L20_L26_sizes, L20_L26_tanh, L20_L26_smap = self.a4(x)

        x, L27_L33_zeros, L27_L33_sizes, L27_L33_tanh, L27_L33_smap = self.b4(x)

        x, L34_L40_zeros, L34_L40_sizes, L34_L40_tanh, L34_L40_smap = self.c4(x)
        
        x, L41_L47_zeros, L41_L47_sizes, L41_L47_tanh, L41_L47_smap = self.d4(x)

        x, L48_L54_zeros, L48_L54_sizes, L48_L54_tanh, L48_L54_smap = self.e4(x)

        x = self.maxpool(x)
        L55_zeros, L55_sizes = sparsity_rate(x)
        L55_tanh = tanh(x, self.beta)
        L55_smap = sparsity_map(x)

        x, L56_L62_zeros, L56_L62_sizes, L56_L62_tanh, L56_L62_smap = self.a5(x)

        x, L63_L69_zeros, L63_L69_sizes, L63_L69_tanh, L63_L69_smap = self.b5(x)

        x = self.avgpool(x)
        L70_zeros, L70_sizes = sparsity_rate(x)
        L70_tanh = tanh(x, self.beta)
        L70_smap = sparsity_map(x)

        x = self.dropout(x)
        L71_zeros, L71_sizes = sparsity_rate(x)
        L71_tanh = tanh(x, self.beta)
        L71_smap = sparsity_map(x)
        
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        # Adds all the tanh outputs, negates them, and finally divides them by number_of_neurons in the network
        tanh_total = (L1_tanh + L2_tanh + L3_tanh + L4_tanh + sum(L5_L11_tanh) + sum(L12_L18_tanh) + L19_tanh + sum(L20_L26_tanh) + sum(L27_L33_tanh) + sum(L34_L40_tanh) + sum(L41_L47_tanh) + sum(L48_L54_tanh) + L55_tanh + sum(L56_L62_tanh) + sum(L63_L69_tanh) + L70_tanh + L71_tanh)
        tanh_total = (-tanh_total)/(L1_sizes + L2_sizes + L3_sizes + L4_sizes + sum(L5_L11_sizes) + sum(L12_L18_sizes) + L19_sizes + sum(L20_L26_sizes) + sum(L27_L33_sizes) + sum(L34_L40_sizes) + sum(L41_L47_sizes) + sum(L48_L54_sizes) + L55_sizes + sum(L56_L62_sizes) + sum(L63_L69_sizes) + L70_sizes + L71_sizes)

        # Concatenate the gathered flattened tensors to generate a unified sparsity-map for entire inference
        L1_smap_flat = L1_smap.view(-1)
        L2_smap_flat = L2_smap.view(-1)
        L3_smap_flat = L3_smap.view(-1)
        L4_smap_flat = L4_smap.view(-1)
        L5_L11_smap_flat  = [tensor.view(-1) for tensor in L5_L11_smap]
        L12_L18_smap_flat = [tensor.view(-1) for tensor in L12_L18_smap]
        L19_smap_flat = L19_smap.view(-1)
        L20_L26_smap_flat = [tensor.view(-1) for tensor in L20_L26_smap]
        L27_L33_smap_flat = [tensor.view(-1) for tensor in L27_L33_smap]
        L34_L40_smap_flat = [tensor.view(-1) for tensor in L34_L40_smap]
        L41_L47_smap_flat = [tensor.view(-1) for tensor in L41_L47_smap]
        L48_L54_smap_flat = [tensor.view(-1) for tensor in L48_L54_smap]
        L55_smap_flat = L55_smap.view(-1)
        L56_L62_smap_flat = [tensor.view(-1) for tensor in L56_L62_smap]
        L63_L69_smap_flat = [tensor.view(-1) for tensor in L63_L69_smap]
        L70_smap_flat = L70_smap.view(-1)
        L71_smap_flat = L71_smap.view(-1)
        
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat] + L5_L11_smap_flat + L12_L18_smap_flat + [L19_smap_flat]
                                       + L20_L26_smap_flat + L27_L33_smap_flat + L34_L40_smap_flat + L41_L47_smap_flat + L48_L54_smap_flat + [L55_smap_flat]
                                       + L56_L62_smap_flat + L63_L69_smap_flat + [L70_smap_flat] + [L71_smap_flat], dim=0)
        
        zeros_list = [L1_zeros] + [L2_zeros] + [L3_zeros] + [L4_zeros] + L5_L11_zeros + L12_L18_zeros + [L19_zeros] + L20_L26_zeros + L27_L33_zeros + L34_L40_zeros + L41_L47_zeros + L48_L54_zeros + [L55_zeros] + L56_L62_zeros + L63_L69_zeros + [L70_zeros] + [L71_zeros]
        sizes_list = [L1_sizes] + [L2_sizes] + [L3_sizes] + [L4_sizes] + L5_L11_sizes + L12_L18_sizes + [L19_sizes] + L20_L26_sizes + L27_L33_sizes + L34_L40_sizes + L41_L47_sizes + L48_L54_sizes + [L55_sizes] + L56_L62_sizes + L63_L69_sizes + [L70_sizes] + [L71_sizes]

        return x, zeros_list, sizes_list, tanh_total, sparsity_map_total
    
def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activations = input_tensor.numel()
    return zeros, activations

# To compute tanh with beta parameter getting input_tensor
def tanh(input_tensor, beta):
    # Define beta value
    BETA = beta
    # Scale the tensor by BETA
    scaled_tensor = BETA * input_tensor
    # Apply tanh function to the scaled tensor
    output = torch.tanh(scaled_tensor)
    # Sum the activations of the tensor
    sum_of_activations = output.sum()
    return sum_of_activations

# To compute Sparsity-Map for each layer's output
def sparsity_map(input_tensor):
    # To create a tensor with all zeros and with the same shape of input tensor
    sparsity_map = torch.zeros_like(input_tensor, dtype=torch.uint8)
    # To find the positions where the input tensor is zero
    zero_positions = torch.eq(input_tensor, 0)
    # To set the corresponding positions in the bit_map to 1
    sparsity_map = sparsity_map.masked_fill(zero_positions, 1)
    return sparsity_map

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]
def clip_tensor(input_tensor, eps):
    tensor_norm = torch.norm(input_tensor, p=2)
    max_norm = eps

    # If the L2 norm is greater than the eps, we scale down the tensor by multiplying it with a factor of (max_norm/tensor_norm)
    # To ensure its norm does not exceed eps. This scaling ensures that the L2 norm of the clipped tensor will be equal to eps.
    if tensor_norm > max_norm:
        clipped_tensor = input_tensor * (max_norm / tensor_norm)
    else:
        clipped_tensor = input_tensor

    return torch.clamp(clipped_tensor, 0, 1)

def calculate_range(current_sparsity_list, sparsity_range, class_index):
    
    for layer in range (len(current_sparsity_list)):

        if current_sparsity_list[layer] < sparsity_range[class_index][layer][0]:
            sparsity_range[class_index][layer][0] = current_sparsity_list[layer]

        if current_sparsity_list[layer] > sparsity_range[class_index][layer][1]:
            sparsity_range[class_index][layer][1] = current_sparsity_list[layer]

    return sparsity_range

# The output ranges from 0 to 1, where 0 means no similarity and 1 means identical tensors (all non-zero elements are the same)
def jaccard_similarity(tensor1, tensor2):
    intersection = torch.logical_and(tensor1, tensor2).sum().item()
    union = torch.logical_or(tensor1, tensor2).sum().item()
    return intersection / union

# Generates sparsity attack for P2=50% of each class in test_dataset
def Sparsity_Attack(model, device, test_loader, num_classes, c_init, args):
    
    coeff = c_init
    correct_after = 0
    correct_before = 0
    total_net_zeros_before = 0
    total_net_sizes_before = 0
    total_net_zeros_after  = 0
    total_net_sizes_after  = 0
    L2_Norms = []

    # A value of 1 means the sparsity attack has polluted the last processed input data of the same class, so we need to leave the current input clean.
    # A value of 0 means the sparsity attack has left the last processed input data of the same class clean, so we need to pollute the current input.
    last_status_of_class  = [0] * num_classes
    num_of_items_in_class = [0] * num_classes
    num_of_adver_in_class = [0] * num_classes
    adversarial_data = []

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):

        data, target = data.to(device), target.to(device)
        i_max = args.imax
        c_min = 0
        c_max = 1
        eps = args.eps
        eps_iter = args.eps_iter

        output = model(data)
        net_zeros = sum(output[1])
        net_sizes = sum(output[2])
        total_net_zeros_before += net_zeros
        total_net_sizes_before += net_sizes
        init_pred = output[0].max(1, keepdim=False)[1] # To get the index of the maximum log-probability for Clean Inputs
        correct_before += (init_pred == target).sum().item()

        x = data
        for i in range(i_max):
            # Sets requires_grad attribute of X to True to compute gradients with respect to the input data
            x = x.clone().detach().requires_grad_(True)

            output = model(x)
            
            if (i>0):
                final_pred = output[0].max(1, keepdim=False)[1]
                if final_pred.item() != init_pred.item():
                    coeff = (coeff + c_max)/2
                else:
                    coeff = (coeff + c_min)/2
            
            l_ce = F.cross_entropy(output[0], target)
            l_sparsity = output[3] # To compute SR considering Tanh function
            l_x = l_sparsity + (coeff * l_ce)
                    
            optimizer.zero_grad()
            l_x.backward()
                
            # To compute gradient of Loss function (l_x) w.r.t input x
            # Below line is used instead of g(i) = μg(i−1) + x.grad.data, because optimizer has already been set as SGD with momentum of 0.9
            g = x.grad.data
            x_new = x - (eps_iter * (g/torch.norm(g, p=2)))
            if args.constrained:
                x_new = clip_tensor(x_new, eps)
            x = x_new

        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of P2=50%
        if last_status_of_class[target.item()] == 0:
            if args.store_attack:
                adversarial_data.append((x_new, target, 1))
            last_status_of_class[target.item()] = 1
            num_of_items_in_class[target.item()] = num_of_items_in_class[target.item()] + 1
            num_of_adver_in_class[target.item()] = num_of_adver_in_class[target.item()] + 1
        else:
            if args.store_attack:
                adversarial_data.append((data, target, 0)) 
            last_status_of_class[target.item()] = 0
            num_of_items_in_class[target.item()] = num_of_items_in_class[target.item()] + 1

        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-data), p=2)
        L2_Norms.append(l2norm_diff)

        # Final prediction after generating attack (The last x_new has remained and needs to be fed into the model)
        output = model(x_new)
        final_pred = output[0].max(1, keepdim=True)[1]
        correct_after += (final_pred == target).sum().item()

        # Re-compute the sparsity rate using the perturbed input
        net_zeros = sum(output[1])
        net_sizes = sum(output[2])
        total_net_zeros_after += net_zeros
        total_net_sizes_after += net_sizes

    # To Create a new dataset using the AdversarialDataset class
    adversarial_dataset = AdversarialDataset(adversarial_data)
    print(f"Distribution of data for each class: {num_of_items_in_class}")
    print(f"Number of generated adversarials in each class: {num_of_adver_in_class}")

    # Calculate overal accuracy of all test data after sparsity attack
    initial_acc = correct_before/float(len(test_loader))
    final_acc = correct_after/float(len(test_loader))

    return adversarial_dataset, initial_acc, final_acc, L2_Norms, (total_net_zeros_before/total_net_sizes_before), (total_net_zeros_after/total_net_sizes_after)

def Profile (model, device, train_loader, num_classes, args):
    
    updated_maps  = [None] * num_classes
    diff_num_ones = [0] * num_classes
    prev_num_ones = [0] * num_classes
    curr_num_ones = [0] * num_classes
    num_layers = 71
    
    # To define sparsity-range for each class
    range_for_each_class = [[[float('inf'), float('-inf')] for _ in range(num_layers)] for _ in range(num_classes)]

    for index, (data, target) in enumerate(tqdm(train_loader, desc='Data Progress')):
        
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)

        pred = output[0].max(1, keepdim=True)[1]
        if pred.item() == target.item():
            if args.method == 'sparsity-range':
                current_sparsity_rate = [0.0] * num_layers  # Initialize the current_sparsity list reading each model input
                for i in range(num_layers):
                    current_sparsity_rate[i] = output[1][i] / output[2][i]
                range_for_each_class = calculate_range(current_sparsity_rate, range_for_each_class, target.item())

            elif args.method == 'sparsity-map':
                if updated_maps[target.item()] == None:
                    updated_maps[target.item()] = output[4]
                    prev_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item() # Computes the number of 1s in the updated_maps
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                else:
                    updated_maps [target.item()] = torch.bitwise_or(updated_maps[target.item()], output[4]) # output[4] is the Sparsity-Map associated with the current input
                    curr_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()]) # Computes the difference between the number of 1s in the current and previous Sparsity-Maps
                    prev_num_ones[target.item()] = curr_num_ones[target.item()]

                if all(num < 1 for num in diff_num_ones):
                    print(diff_num_ones)
                    break

    return updated_maps, range_for_each_class, index

def Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes):
    
    generated_adversarial_for_class = [0] * num_classes 
    num_of_items_in_class = [0] * num_classes

    # Initialization for sparsity-maps
    sim_threshold = 0.468
    correctly_predicted_adversarial_for_class_map = [0] * num_classes
    correctly_predicted_adversarial_ratio_map = []

    # Initialization for sparsity-ranges
    num_layers = 71 # Assuming 71 Sub-layers for GoogleNet
    layer_inclusion_threshold = num_layers - 70
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    for index, (data, target, adversarial) in enumerate(tqdm(test_dataset, desc='Data Progress')):
        
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)
        
        pred = output[0].max(1, keepdim=True)[1]

        ############################################## sparsity-range #####################################################################

        current_sparsity_rate = [0.0] * num_layers  # To construct the current_sparsity list reading each data input
        for L in range(num_layers):
            current_sparsity_rate[L] = output[1][L] / output[2][L]
        
        in_range_status = [0] * num_layers
        for M in range(num_layers):
            if not offline_sparsity_ranges[pred.item()][M][0] <= current_sparsity_rate[M] <= offline_sparsity_ranges[pred.item()][M][1]:
                in_range_status[M] = 1
        
        if sum(in_range_status) >= layer_inclusion_threshold:
            if adversarial == 1:
                correctly_predicted_adversarial_for_class_range[target.item()] += 1
        
        ############################################## sparsity-map #######################################################################
        
        sim_rate = jaccard_similarity(offline_sparsity_maps[pred.item()], output[4])

        # If sim_rate was not more than a specific threshold (i.e., sim_threshold), we predict that the input is adversarial
        if sim_rate <= sim_threshold:
            if adversarial == 1:
                correctly_predicted_adversarial_for_class_map[target.item()] += 1

        # To check the real adversarial status for the same targeted class
        if adversarial == 1:
            generated_adversarial_for_class[target.item()] += 1 

        num_of_items_in_class[target.item()] += 1

        ###################################################################################################################################

    for predicted, generated in zip(correctly_predicted_adversarial_for_class_map, generated_adversarial_for_class):
        correctly_predicted_adversarial_ratio_map.append((predicted/generated)*100)

    for predicted, generated in zip(correctly_predicted_adversarial_for_class_range, generated_adversarial_for_class):
        correctly_predicted_adversarial_ratio_range.append((predicted/generated)*100)

    correctly_predicted_adversarial_ratio_map = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_map]

    correctly_predicted_adversarial_ratio_range = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_range]

    overall_accuracy_map = sum(correctly_predicted_adversarial_for_class_map)/sum(generated_adversarial_for_class)
    overall_accuracy_range = sum(correctly_predicted_adversarial_for_class_range)/sum(generated_adversarial_for_class)
    
    print(f"\nDistribution of data in each class: {num_of_items_in_class}\n")
    print(f"Correctly predicted adversarials for each class (map): {correctly_predicted_adversarial_for_class_map}\n")
    print(f"Correctly predicted adversarials for each class (rng): {correctly_predicted_adversarial_for_class_range}\n")
    print(f"Number of generated adversarials for each class: {generated_adversarial_for_class}\n")
    print(f"Percentage of correctly predicted adversarials for each class (map): {correctly_predicted_adversarial_ratio_map}\n")
    print(f"Percentage of correctly predicted adversarials for each class (rng): {correctly_predicted_adversarial_ratio_range}\n")
    print(f"Overall attack detection accuracy using sparsity-map method: {overall_accuracy_map*100:.2f}\n")
    print(f"Overall attack detection accuracy using sparsity-range method: {overall_accuracy_range*100:.2f}\n")

    return

if __name__ == '__main__':

    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file you want to test')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial training learning rate')
    parser.add_argument('--phase', default='test', help="test, profile, sparsity-attack, and sparsity-detect. Training can be performed using train.py")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism based on sparsity-map or sparsity-range")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon in each inner loop iteration")
    parser.add_argument('--constrained', action='store_true', help="To enable clipping the generated purturbed data")
    parser.add_argument('--store_attack', action='store_true', help="To lead the algorithm to store the generated adversarials")
    args = parser.parse_args()
    print(args)

    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n A {device} is assigned for processing! \n")

    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    if args.phase == 'test':
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=TRANSFORM)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.phase == 'sparsity-attack':
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=TRANSFORM)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    elif args.phase == 'profile':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=TRANSFORM)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    elif args.phase == 'sparsity-detect':
        # To load the adversarial_dataset from the offline file generated by sparsity-attack function (50% are polluted)
        test_dataset = torch.load('./adversarial_data/adversarial_dataset_constrained.pt', map_location=device)

    # Network Initialization
    model = GoogleNet(beta=args.beta)
    model.to(device)

    # Network Configuration
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.phase == 'test':

        if args.weights is not None:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(args.weights))
            else:
                model.load_state_dict(torch.load(args.weights, map_location=device))

        correct = 0
        total = 0
        net_zeros_total = 0
        net_sizes_total = 0
        zeros_total = [0] * 71
        sizes_total = [0] * 71

        model.eval()
        with torch.no_grad():
            for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):
                image, target = data.to(device), target.to(device)
                outputs = model(image)

                # Sparsity calculations zone begin
                zeros_total = [xx + yy for xx, yy in zip(zeros_total, outputs[1])]
                sizes_total = [xx + yy for xx, yy in zip(sizes_total, outputs[2])]
                net_zeros_total += sum(outputs[1])
                net_sizes_total += sum(outputs[2])
                # Sparsity calculations zone end

                # Compute accuracy
                _, predicted = torch.max(outputs[0].data, 1) # To find the index of max-probability for each output in the BATCH
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print()
        print("Network accuracy on 10000 test images: ", (correct / len(test_loader.dataset))*100)
        print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))

        # Calculating average sparsity rate for each layer (after processing whole testset)
        SR = []
        for i in range(71):
            SR.append(zeros_total[i] / sizes_total[i])
        
        # Calculating average sparsity rate for entire network (after processing whole testset)
        SR_Net = net_zeros_total/net_sizes_total

        # Printing Sparsity Statistics
        print()
        for i in range(71):
            print(f"SR_L{i+1}: {SR[i]}")
        print(f"SR_Net: {SR_Net}")

    elif args.phase == 'sparsity-attack':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        num_classes = len(test_dataset.classes)

        c_init = 0.5
        adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total\
            = Sparsity_Attack(model, device, test_loader, num_classes, c_init, args)
        
        # Save the generated adversarial dataset to disk
        torch.save(adversarial_dataset, './adversarial_data/adversarial_dataset.pt')

        print(f"Test accuracy excluding energy attack: {initial_accuracy}")
        print(f"Test accuracy including energy attack: {final_accuracy}")
        print(f"Sparsity rate before energy attack: {sr_net_before_total}")
        print(f"Sparsity rate after energy attack: {sr_net_after_total}")
        print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
        print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")

    elif args.phase == 'profile':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()
        
        num_classes  = len(train_dataset.classes)

        # The profile function returns an array in which each element comprises a Sparsity-Map Tensor assigned to each Class
        sparsity_maps, sparsity_range, index = Profile(model, device, train_loader, num_classes, args)
        
        # Store the Sparsity_Maps in an offline file
        print(f"We should stop profiling at data-point {index} when profiling is based on {args.method}")
        print(f"P1 = {index/(len(train_loader)-1)*100:.2f} % of training-set has been used for profiling.")

        if args.method == 'sparsity-map':
            # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_maps, "./adversarial_data/sparsity_maps.pt", pickle_protocol=4)
        elif args.method == 'sparsity-range':
            # Save the Sparsity Range to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_range, "./adversarial_data/sparsity_ranges.pth", pickle_protocol=4)

    elif args.phase == 'sparsity-detect':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        # Load the sparsity-maps and sparsity-ranges from offline file (generated by profiling trainingset)
        offline_sparsity_maps = torch.load("./adversarial_data/sparsity_maps.pt")
        offline_sparsity_ranges = torch.load("./adversarial_data/sparsity_ranges.pth")
        
        num_classes  = len(offline_sparsity_maps) # or length of (offline_sparsity_ranges)

        # Prints the number of detected adversarials in each class
        Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes)    

# Test:    python3 main_single_loop.py --phase test --weights checkpoint/googlenet-169-best.pth
# Attack:  python3 main_single_loop.py --phase sparsity-attack --store_attack --constrained --imax 220 --beta 20 --eps 0.99 --eps_iter 0.2 --weights checkpoint/googlenet-169-best.pth
# Profile: python3 main_single_loop.py --phase profile --method sparsity-map/range --weights checkpoint/googlenet-169-best.pth
# Detect:  python3 main_single_loop.py --phase sparsity-detect --weights checkpoint/googlenet-169-best.pth
