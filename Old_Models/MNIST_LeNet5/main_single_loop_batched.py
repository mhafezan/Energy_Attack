import math
import time
import torch
import argparse
import os
import sys
import numpy as np
from torch import nn
from torch.nn import Module
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Custom Dataset to store the adversarial data
class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# LeNet5 Model definition
class LeNet5(Module):
    def __init__(self, args):
        super(LeNet5, self).__init__()
        self.beta  = args.beta
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

    def forward(self, x, SR=True, PC=False):
        
        L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size = (0,) * 10
        
        pow_dnmc = []
        pow_stat = []
        num_cycles = []
        latency = []
        
        if PC:
            dnmc, stat, cycles = conv_power(x, self.conv1.weight, self.conv1.stride, self.conv1.padding, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L1_zeros, L1_size = sparsity_rate(x)
        L1_elements = x[0].numel()
        L1_tanh = tanh(x, self.beta)
        L1_smap = sparsity_map(x)
        L1_start = time.time()
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        L1_end = time.time()
        latency.append(L1_end - L1_start)

        if PC:
            dnmc, stat, cycles = conv_power(x, self.conv2.weight, self.conv2.stride, self.conv2.padding, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L2_zeros, L2_size = sparsity_rate(x)
        L2_elements = x[0].numel()
        L2_tanh = tanh(x, self.beta)
        L2_smap = sparsity_map(x)
        L2_start = time.time()
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        L2_end = time.time()
        latency.append(L2_end - L2_start)

        x = x.view(x.shape[0], -1)

        if PC:
            dnmc, stat, cycles = fc_power(x, self.fc1.weight, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L3_zeros, L3_size = sparsity_rate(x)
        L3_elements = x[0].numel()
        L3_tanh = tanh(x, self.beta)
        L3_smap = sparsity_map(x)
        L3_start = time.time()
        x = self.fc1(x)
        x = self.relu3(x)
        L3_end = time.time()
        latency.append(L3_end - L3_start)

        if PC:
            dnmc, stat, cycles = fc_power(x, self.fc2.weight, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L4_zeros, L4_size = sparsity_rate(x)
        L4_elements = x[0].numel()
        L4_tanh = tanh(x, self.beta)
        L4_smap = sparsity_map(x)
        L4_start = time.time()
        x = self.fc2(x)
        x = self.relu4(x)
        L4_end = time.time()
        latency.append(L4_end - L4_start)

        if PC:
            dnmc, stat, cycles = fc_power(x, self.fc3.weight, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L5_zeros, L5_size = sparsity_rate(x)
        L5_elements = x[0].numel()
        L5_tanh = tanh(x, self.beta)
        L5_smap = sparsity_map(x)
        L5_start = time.time()
        x = self.fc3(x)
        x = self.relu5(x)
        L5_end = time.time()
        latency.append(L5_end - L5_start)

        # Adds all the tanh outputs, negates them, and finally divides them by number_of_neurons of network (-tanh_total indicates the number of zeros)
        tanh_total = (L1_tanh + L2_tanh + L3_tanh + L4_tanh + L5_tanh)
        tanh_total = (-tanh_total)/(L1_elements + L2_elements + L3_elements + L4_elements + L5_elements)

        # Concatenate the flattened tensors to generate a Unified Sparsity-Map
        L1_smap_flat = L1_smap.view(-1)
        L2_smap_flat = L2_smap.view(-1)
        L3_smap_flat = L3_smap.view(-1)
        L4_smap_flat = L4_smap.view(-1)
        L5_smap_flat = L5_smap.view(-1)
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat, L5_smap_flat], dim=0)
        
        zeros_list = [L1_zeros] + [L2_zeros] + [L3_zeros] + [L4_zeros] + [L5_zeros]
        sizes_list = [L1_size]  + [L2_size]  + [L3_size]  + [L4_size]  + [L5_size]

        return x, zeros_list, sizes_list, tanh_total, sum(pow_dnmc), sum(pow_stat), sum(num_cycles), sum(latency), sparsity_map_total

def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

# To compute tanh with beta parameter
def tanh(input_tensor, beta):
    # To scale the tensor by BETA and apply tanh function to the scaled tensor
    output = torch.tanh(beta * input_tensor)
    # To sum the activations separately for each image in the batch
    return output.view(input_tensor.size(0), -1).sum(dim=1)

# To compute Sparsity-Map for each layer's output
def sparsity_map(input_tensor):
    # To create a tensor with all zeros and with the same shape of input tensor
    maps = torch.zeros_like(input_tensor, dtype=torch.uint8)
    # To find the positions where the input tensor is zero
    zero_positions = torch.eq(input_tensor, 0)
    # To set the corresponding positions in the bit_map to 1
    maps = maps.masked_fill(zero_positions, 1)
    return maps

def conv_power (input, weight, stride, padding, architecture):

    pow_dynamic = None
    pow_static = None
    total_cycles = None
    
    num_all_mul = 0
    num_non_zero_mul = 0
    num_input_elements = input.numel()
    num_non_zero_inputs = torch.count_nonzero(input).item()
    num_output_elements = input.size(0) * weight.size(0) * int((input.size(2) + (2 * padding[0]) - weight.size(2) + 1)/stride[0]) * int((input.size(3) + (2 * padding[1]) - weight.size(3) + 1)/stride[1])
            
    # Iterate over each element according to the convolution functionality
    for b in range(input.size(0)): # Batch dimension
        for c_out in range(weight.size(0)): # Output channel dimension (to iterate over different kernels)
            for h_out in range(0, input.size(2) + (2 * padding[0]) - weight.size(2) + 1, stride[0]): # Output Height dimension
                for w_out in range(0, input.size(3) + (2 * padding[1]) - weight.size(3) + 1, stride[1]): # Output Width dimension
                    for c_in in range(input.size(1)): # Input/Kernel channel dimension
                        for h_k in range(weight.size(2)): # Kernel height dimension
                            for w_k in range(weight.size(3)): # Kernel width dimension
                                num_all_mul += 1
                                input_operand = input[b, c_in, h_out + h_k, w_out + w_k]
                                if input_operand.item() != 0: # As weight pruning is not done in CNVLUTIN, we avoid checking the weight operands
                                    num_non_zero_mul += 1
    
    if (architecture == "dadiannao"):
        num_non_zero_mul = num_all_mul
    
    pow_dnmc_multipliers = num_non_zero_mul*0.000414521521 # Number of accesses to all multipliers is equal to the non-zero multiplications (W)
    pow_stat_multipliers = 0.061512900608 # 16*256 multipliers are accessed within a node in each cycle: (16*256)*0.000015017798 (W)
    cycles_multipliers = math.ceil(num_non_zero_mul/4096) # Required cycles to perform all non-zero multiplications in parallel: non_zero_mul/(16*256)
            
    pow_dnmc_adders = num_non_zero_mul*0.000383151011 # Number of accessed adders: (num_non_zero_mul/16)*0.006130416161 (W)
    pow_stat_adders = 0.032082976 # Number of accessed adders in each cycle: (16*16)*0.000125324125 (W)
    cycles_adders = 0 # It has overlap with NFU cycles
            
    pow_dnmc_encoder = num_output_elements*0.000036327811 # Number of accesses to all encoders within a node: (num_output_elements/16)*0.000581244983 (W)
    pow_stat_encoder = 0.00101210944 # Number of accessed encoders in each cycle: 16*0.00006325684 (W)
    cycles_encoder = math.ceil(num_output_elements/16) # Number of cycles to encode all outputs by 16 parallel encoders in 16 cycles: (num_output_elements/256)*16
            
    # The number of accesses to Dispatcher is equal to the number of bricks in NM (or Input Tensor), as each brick in each lane should be proccessed
    # separately by one of 16 parallel lanes in the Dispatcher. Each Brick Buffer (BB) Entry in the Dispatcher is connected to corresponding NM Bank
    # via a 256-bit BUS (16-neuron-wide) and is able to read 256 bits (i.e., one brick) within 4 cycles (NM delay).
    pow_dnmc_dispatcher = num_input_elements*0.000479077231187 # (num_input_elements/16)*0.007665235699 (W)
    pow_stat_dispatcher = 0.000511910231 # (W)
    cycles_dispatcher = math.ceil(num_non_zero_inputs/16) # If 16 non-zero inputs are detected simultaneously in all Dispatcher lanes over a cycle, one
    # shift to the output is done, otherwise Dispatcher is clock-gated. Accordingly, we devide the required cycles by 16 (Paper: Dispatcher broadcasts
    # non-zero neurons, a single-neuron from each BB entry at a time, for a total of 16 neurons, one per BB entry and thus per neuron lane each cycle).
            
    pow_dnmc_relu = (num_output_elements)*0.00000186564 # ReLU component is activated once after each output element generation (W)
    pow_stat_relu = 0.000060372 # 256*0.00000023583: 16 non-zeros are dispatched, 256 outputs are generated in parallel, and activated by 256 ReLUs (W)
    cycles_relu = 0 # It has overlap with NFU cycles
            
    pow_dnmc_nbin = num_non_zero_inputs*(0.000000000046322768) # ((num_non_zero_inputs/16)*256)*(Read+Write Energy) (W)
    pow_stat_nbin = 0.182776576 # 256*0.000713971 (W)
    cycles_nbin = math.ceil(num_non_zero_inputs/16) # Cycles to Read and Write from/into 256 parallel Nbins (Write is covered by Dispatcher cycles)
            
    pow_dnmc_offset = pow_dnmc_nbin
    pow_stat_offset = pow_stat_nbin
    cycles_offset = 0 # It has overlap with nbin cycles (Paper: Every cycle, each subunit fetches a single (neuron, offset) pair from NBin, uses the
    # offset to index the corresponding entry from its SBin to fetch 16 synapses and produces 16 products, one per filter)
            
    pow_dnmc_SB = (num_non_zero_inputs*0.00000000005082784) # 16 entries of one SB are accessed in 1 cycle: ((num_non_zero_inputs/16)*256)*(0.00000000000317674)
    pow_stat_SB = 0.00131820288 # 256*0.00000514923 (W)
    cycles_SB = 0 # It has overlap with nbin cycles (Paper: we decided to clock the NFU at the same frequency as the eDRAM available in 28nm technology
    # Thus SB cycles is covered by one processor cycle)
            
    pow_dnmc_nbout = (num_non_zero_mul*0.000000000000011755585938)+(num_output_elements+num_non_zero_mul)*(0.000000000000114434375) # (num_non_zero_mul/256)*(0.00300943)+(num_output_elements/16+num_non_zero_mul/16)*(0.00183095) (W)
    pow_stat_nbout = 0.04516752 # 16 * 0.00282297 (W)
    cycles_nbout = 0 # Read Cycles are covered by encoder cycles and Write Cycles are covered by NFU Cycles.
            
    pow_dnmc_nm = (num_input_elements*0.0000000000016507625)+(num_output_elements*0.00000000004045925) # (num_input_elements/16)*(0.0264122)+(num_output_elements/16)*(0.647348) (W)
    pow_stat_nm = 0.0132751 # (W)
    cycles_nm = math.ceil(num_input_elements/16) # X 1 cycle. Number of cycles required to read all bricks from NM by Dispatcher (Write cycles overlaps
    # with NFU cycles). The NM latency for DaDianNao architecture is 10-cycles, where NFU and eDRAM are clocked at 606 MHz in 28nm technology. As NM
    # cycle time (from CACTI) in 45nm is 7.32542 nS, maintaining 606 MHz frequency for the NFU, we can compute the NM latency for 45nm as follows:
    # (7.32542/2)=3.66271 ~ 4 cycles. However, our designed NFU critical path delay is 20.296523 ns, thus, (7.32542/20.296523)=0.360919947 ~ 1 cycle
            
    pow_dynamic = pow_dnmc_multipliers + pow_dnmc_adders + pow_dnmc_encoder + pow_dnmc_dispatcher + pow_dnmc_nbin + pow_dnmc_offset + pow_dnmc_SB + pow_dnmc_nbout + pow_dnmc_nm + pow_dnmc_relu
                       
    pow_static = pow_stat_multipliers + pow_stat_adders + pow_stat_encoder + pow_stat_dispatcher + pow_stat_nbin + pow_stat_offset + pow_stat_SB + pow_stat_nbout + pow_stat_nm + pow_stat_relu
                         
    total_cycles = cycles_multipliers + cycles_adders + cycles_encoder + cycles_dispatcher + cycles_nbin + cycles_offset + cycles_SB + cycles_nbout + cycles_nm + cycles_relu

    return pow_dynamic, pow_static, total_cycles

def fc_power (input, weight, architecture):

    pow_dynamic = None
    pow_static = None
    total_cycles = None
    
    num_all_mul = 0
    num_non_zero_mul = 0
    num_input_elements = input.numel()
    num_non_zero_inputs = torch.count_nonzero(input).item()
    num_output_elements = input.size(0) * weight.size(0)
            
    # Iterate over each element of the input tensor and the weight tensor to compute the number of non-zero multiplications
    for i in range(input.size(0)): # Iterate over batch dimension
        for j in range(weight.size(0)): # Iterate over output dimension
            for k in range(input.size(1)): # Iterate over the identical-size dimension
                num_all_mul += 1
                if input[i, k].item() != 0:
                    num_non_zero_mul += 1

    if (architecture == "dadiannao"):
        num_non_zero_mul = num_all_mul
            
    pow_dnmc_multipliers = num_non_zero_mul*0.000414521521 # Number of accesses to all multipliers is equal to the non-zero multiplications (W)
    pow_stat_multipliers = 0.061512900608 # 16*256 Multipliers are accessed within a node in each cycle: (16*256)*0.000015017798 (W)
    cycles_multipliers = math.ceil(num_non_zero_mul/4096) # Required cycles to perform all non-zero multiplications in parallel: non_zero_mul/(16*256)
            
    pow_dnmc_adders = num_non_zero_mul*0.000383151011 # Number of accesses to adders: (num_non_zero_mul/16)*0.006130416161 (W)
    pow_stat_adders = 0.032082976 # Number of accessed adders in each cycle: (16*16)*0.000125324125 (W)
    cycles_adders = 0 # It has overlap with NFU cycles
            
    pow_dnmc_encoder = num_output_elements*0.000036327811 # Number of accesses to all encoders within a node: (num_output_elements/16)*0.000581244983 (W)
    pow_stat_encoder = 0.00101210944 # Number of accessed encoders in each cycle: 16*0.00006325684 (W)
    cycles_encoder = math.ceil(num_output_elements/16) # Number of cycles to encode all outputs by 16 parallel encoders in 16 cycles: (num_output_elements/256)*16
            
    # The number of accesses to Dispatcher is equal to the number of bricks in NM (or Input Tensor), as each brick in each lane should be proccessed
    # separately by one of 16 parallel lanes in the Dispatcher. Each Brick Buffer (BB) Entry in the Dispatcher is connected to corresponding NM Bank
    # via a 256-bit BUS (16-neuron-wide) and is able to read 256 bits (i.e., one brick) within 4 cycles (NM delay).
    pow_dnmc_dispatcher = num_input_elements*0.000479077231187 # (num_input_elements/16)*0.007665235699 (W)
    pow_stat_dispatcher = 0.000511910231 # (W)
    cycles_dispatcher = math.ceil(num_non_zero_inputs/16) # If 16 non-zero inputs are detected simultaneously in all Dispatcher lanes over a cycle, one
    # shift to the output is done, otherwise Dispatcher is clock-gated. Accordingly, we devide the required cycles by 16 (Paper: Dispatcher broadcasts
    # non-zero neurons, a single-neuron from each BB entry at a time, for a total of 16 neurons, one per BB entry and thus per neuron lane each cycle).
            
    pow_dnmc_relu = (num_output_elements)*0.00000186564 # ReLU component is activated and accessed once after each output element generation (W)
    pow_stat_relu = 0.000060372 # 256*0.00000023583: 16 non-zeros are dispatched, 256 outputs are generated in parallel, and activated by 256 ReLUs (W)
    cycles_relu = 0 # It has overlap with NFU cycles
            
    pow_dnmc_nbin = num_non_zero_inputs*(0.000000000046322768) # ((num_non_zero_inputs/16)*256)*(Read+Write Energy) (W)
    pow_stat_nbin = 0.182776576 # 256*0.000713971 (W)
    cycles_nbin = math.ceil(num_non_zero_inputs/16) # Cycles to Read and Write from/into 256 parallel Nbins (Write is covered by Dispatcher cycles)
            
    pow_dnmc_offset = pow_dnmc_nbin
    pow_stat_offset = pow_stat_nbin
    cycles_offset = 0 # It has overlap with nbin cycles (Paper: Every cycle, each subunit fetches a single (neuron, offset) pair from NBin, uses the
    # offset to index the corresponding entry from its SBin to fetch 16 synapses and produces 16 products, one per filter)
            
    pow_dnmc_SB = (num_non_zero_inputs*0.00000000005082784) # 16 entries of one SB are accessed in 1 cycle: ((num_non_zero_inputs/16)*256)*(0.00000000000317674)
    pow_stat_SB = 0.00131820288 # 256*0.00000514923 (W)
    cycles_SB = 0 # It has overlap with nbin cycles (Paper: we decided to clock the NFU at the same frequency as the eDRAM available in 28nm technology
    # Thus SB cycles is covered by one processor cycle)
            
    pow_dnmc_nbout = (num_non_zero_mul*0.000000000000011755585938)+(num_output_elements+num_non_zero_mul)*(0.000000000000114434375) # (num_non_zero_mul/256)*(0.00300943)+(num_output_elements/16+num_non_zero_mul/16)*(0.00183095) (W)
    pow_stat_nbout = 0.04516752 # 16 * 0.00282297 (W)
    cycles_nbout = 0 # Read Cycles are covered by encoder cycles and Write Cycles are covered by NFU Cycles.
            
    pow_dnmc_nm = (num_input_elements*0.0000000000016507625)+(num_output_elements*0.00000000004045925) # (num_input_elements/16)*(0.0264122)+(num_output_elements/16)*(0.647348) (W)
    pow_stat_nm = 0.0132751 # (W)
    cycles_nm = math.ceil(num_input_elements/16) # X 1 cycle. Number of cycles required to read all bricks from NM by Dispatcher (Write cycles overlaps
    # with NFU cycles). The NM latency for DaDianNao architecture is 10-cycles, where NFU and eDRAM are clocked at 606 MHz in 28nm technology. As NM
    # cycle time (from CACTI) in 45nm is 7.32542 nS, maintaining 606 MHz frequency for the NFU, we can compute the NM latency for 45nm as follows:
    # (7.32542/2)=3.66271 ~ 4 cycles. However, our designed NFU critical path delay is 20.296523 ns, thus, (7.32542/20.296523)=0.360919947 ~ 1 cycle
            
    pow_dynamic = pow_dnmc_multipliers + pow_dnmc_adders + pow_dnmc_encoder + pow_dnmc_dispatcher + pow_dnmc_nbin + pow_dnmc_offset + pow_dnmc_SB + pow_dnmc_nbout + pow_dnmc_nm + pow_dnmc_relu
                       
    pow_static = pow_stat_multipliers + pow_stat_adders + pow_stat_encoder + pow_stat_dispatcher + pow_stat_nbin + pow_stat_offset + pow_stat_SB + pow_stat_nbout + pow_stat_nm + pow_stat_relu
                         
    total_cycles = cycles_multipliers + cycles_adders + cycles_encoder + cycles_dispatcher + cycles_nbin + cycles_offset + cycles_SB + cycles_nbout + cycles_nm + cycles_relu

    return pow_dynamic, pow_static, total_cycles

# FGSM attack function to create adversarial example (perturbed image)
def fgsm_attack(image_clean, epsilon, data_grad):
    """
    data_grad is the gradient of the loss w.r.t the input image
    """
    sign_data_grad  = data_grad.sign() # Collect the element-wise sign of the data gradient
    perturbed_image = image_clean + epsilon * sign_data_grad # Creates the perturbed image by adjusting each pixel of the input image
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # Clipping perturbed_image within [0,1] range
    return perturbed_image
    
# For each sample in test set, this computes the gradient of the loss w.r.t the input data (data_grad), creates a perturbed image with fgsm_attack (perturbed_data)
# Then checks to see if the perturbed example is adversarial. To test the model accuracy, it saves and returns some successful adversarial examples to be visualized later
def FGSM_Test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor to True to compute gradients with respect to input data
        data.requires_grad = True

        output = model(data, False, False)
        init_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability (mohammad added [0])

        # If the initial prediction (before attack) is wrong, don't bother FGSM to generate adversarial
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output[0], target) # mohammad added [0]

        # Zero all existing gradients and Calculate gradients of model in backward
        model.zero_grad()
        loss.backward()

        # Collect 'data gradients'
        data_grad = data.grad.data

        # Call fgsm_attack function to generate perturbed data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data, False, False)

        # Final prediction after applying attack
        final_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability (mohammad added [0])

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving epsilon=0 examples (attack free)
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # In the case of misprediction, It saves some adv-examples for later visualization
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for the given epsilon after adversarial attack
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# To calculate range
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

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]
def clip_tensor(input_tensor, eps, batch_size):
    
    tensor_norm = torch.stack([torch.norm(input_tensor[i], p=2) for i in range(batch_size)])

    # If the L2 norm of generated adversarial is greater than the eps, we scale it down by a factor of (max_norm / tensor_norm)
    clipped_tensor = torch.stack([torch.where(tensor_norm[i] > eps, input_tensor[i] * (eps / tensor_norm[i]), input_tensor[i]) for i in range(batch_size)])

    max_element, _ = torch.max(clipped_tensor.view(batch_size, -1), dim=1)
    
    # If the maximum element is larger than the 1, we again scale it down by a factor of 1/max_element
    clipped_tensor = torch.stack([torch.where(max_element[i] > 1, clipped_tensor[i].div_(max_element[i]), clipped_tensor[i]) for i in range(batch_size)])
    
    # The default to implement the constrained mode
    # return torch.clamp(clipped_tensor, 0, 1)
    return clipped_tensor
    
# Generates sparsity attack for 50% of images in each class of original testset
def Sparsity_Attack_Generation(model, device, test_loader, num_classes, c_init, args):
    
    num_processed_images = 0
    correct_before = 0
    correct_after = 0
    l2_norms = []
    total_net_zeros_before = 0
    total_net_sizes_before = 0
    total_net_zeros_after = 0
    total_net_sizes_after = 0

    adversarial_data = []
    # A value of 1 means the sparsity attack has polluted the last processed input data of the same class, so we need to leave the current input clean.
    # A value of 0 means the sparsity attack has left the last processed input data of the same class clean, so we need to pollute the current input.
    last_status_of_class  = [0] * num_classes
    num_of_items_in_class = [0] * num_classes
    num_of_adver_in_class = [0] * num_classes

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):

        data, target = data.to(device), target.to(device)
        i_max = args.imax
        c_min = 0
        c_max = 1
        momentum = 0.9
        coeff = torch.full((args.batch_size,), c_init, device=device)
        eps = args.eps
        eps_iter = args.eps_iter

        # To inject the Clean Image to model to compute the accuracy and sparsity rate
        output = model(data, True, False)
        net_zeros = sum(output[1])
        net_sizes = sum(output[2])
        total_net_zeros_before += net_zeros
        total_net_sizes_before += net_sizes
        init_pred = output[0].max(1, keepdim=False)[1]
        correct_before += (init_pred == target).sum().item()

        x = data
        g = torch.zeros_like(x)
        
        for i in range(i_max):

            # To set requires_grad attribute of X to True to compute the gradient of loss function (l_x) w.r.t input X
            x = x.clone().detach().requires_grad_(True)
            
            output = model(x, False, False)
            
            if (i>0):
                final_pred = output[0].max(1, keepdim=False)[1]
                coeff = torch.stack([torch.where(init_pred[i] == final_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
            
            # To compute cross-entropy loss independently for each image in the batch
            l_ce = torch.stack([F.cross_entropy(output[0][j].unsqueeze(0), init_pred[j].unsqueeze(0)) for j in range(args.batch_size)])
            
            # To compute l_sparsity (using Tanh function) independently for each image in the batch
            l_sparsity = output[3]
            
            l_x = l_sparsity + (coeff * l_ce)
                    
            optimizer.zero_grad()
            l_x.backward(torch.ones_like(l_x)) # Using torch.ones_like(l_x) ensures that the gradients of each element in l_x is computed independently 
                
            g = (momentum * g) + x.grad.data
            x_new = torch.stack([x[k] - (eps_iter * (g[k] / torch.norm(g[k], p=2))) for k in range(args.batch_size)])
            if args.constrained:
                x_new = clip_tensor(x_new, eps, args.batch_size)
            x = x_new

        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of 100% for each class
        if args.store_attack:
            adversarial_data.append((x_new, target, 1))
            
        """
        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of 50% for each class
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
        """

        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-data).view(args.batch_size, -1), p=2, dim=1)
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

        # To inject the Adversarial Image to model to compute the accuracy and sparsity rate
        output = model(x_new, True, False)
        final_pred = output[0].max(1, keepdim=False)[1]
        correct_after += (final_pred == target).sum().item()

        # Re-compute the sparsity rate using the perturbed input
        net_zeros = sum(output[1])
        net_sizes = sum(output[2])
        total_net_zeros_after += net_zeros
        total_net_sizes_after += net_sizes

        num_processed_images += args.batch_size

    # To Create a new dataset using the AdversarialDataset class
    adversarial_dataset = AdversarialDataset(adversarial_data)
    print(f"Distribution of data for each class: {num_of_items_in_class}")
    print(f"Number of generated adversarials in each class: {num_of_adver_in_class}")

    # Calculate overall accuracy of all test data after sparsity attack
    first_acc = correct_before/float(num_processed_images)
    final_acc = correct_after/float(num_processed_images)

    return adversarial_dataset, first_acc, final_acc, l2_norms, (total_net_zeros_before/total_net_sizes_before), (total_net_zeros_after/total_net_sizes_after)

def Profile (model, device, train_loader, num_classes, args):
    
    updated_maps  = [None] * num_classes
    diff_num_ones = [0] * num_classes
    prev_num_ones = [0] * num_classes
    curr_num_ones = [0] * num_classes
    num_layers = 5
    
    # To define sparsity-range for each class
    range_for_each_class = [[[float('inf'), float('-inf')] for _ in range(num_layers)] for _ in range(num_classes)]

    for index, (data, target) in enumerate(tqdm(train_loader, desc='Data Progress')):
        
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data, True, False)

        pred = output[0].max(1, keepdim=True)[1]
        if pred.item() == target.item():
            if args.method == 'sparsity-range':
                current_sparsity_rate = [0.0] * num_layers
                for i in range(num_layers):
                    current_sparsity_rate[i] = (output[1][i] / output[2][i]) if output[2][i] != 0 else 0
                range_for_each_class = calculate_range(current_sparsity_rate, range_for_each_class, target.item())

            elif args.method == 'sparsity-map':
                if updated_maps[target.item()] == None:
                    updated_maps[target.item()] = output[8]
                    prev_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                else:
                    updated_maps [target.item()] = torch.bitwise_or(updated_maps[target.item()], output[8])
                    curr_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                    prev_num_ones[target.item()] = curr_num_ones[target.item()]

                if all(num < 1 for num in diff_num_ones):
                    print(diff_num_ones)
                    break

    return updated_maps, range_for_each_class, index

def Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes):
    
    generated_adversarial_for_class = [0] * num_classes 
    num_of_items_in_class = [0] * num_classes

    # Initialization for sparsity-maps
    sim_threshold = 0.323
    correctly_predicted_adversarial_for_class_map = [0] * num_classes
    correctly_predicted_adversarial_ratio_map = []

    # Initialization for sparsity-ranges
    num_layers = 5
    layer_inclusion_threshold = num_layers - 4
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    for index, (data, target, adversarial) in enumerate(tqdm(test_dataset, desc='Data Progress')): 
        
        batch_size = data.size(0)

        for i in range (batch_size):

            single_image = data[i].unsqueeze(0).to(device)
            single_target = target[i].unsqueeze(0).to(device)

            output = model(single_image, True, False)
            
            pred = output[0].max(1, keepdim=True)[1]

            ############################################## sparsity-range #####################################################################

            current_sparsity_rate = [0.0] * num_layers

            for L in range(num_layers):
                current_sparsity_rate[L] = (output[1][L] / output[2][L]) if output[2][L] != 0 else 0
            
            in_range_status = [0] * num_layers
            for M in range(num_layers):
                if not offline_sparsity_ranges[pred.item()][M][0] <= current_sparsity_rate[M] <= offline_sparsity_ranges[pred.item()][M][1]:
                    in_range_status[M] = 1
            
            if sum(in_range_status) >= layer_inclusion_threshold:
                if adversarial == 1:
                    correctly_predicted_adversarial_for_class_range[single_target.item()] += 1

            ############################################## sparsity-map #######################################################################
            
            sim_rate = jaccard_similarity(offline_sparsity_maps[pred.item()], output[8])

            # if the following condition is True, we predict that the input is adversarial
            if sim_rate <= sim_threshold:
                if adversarial == 1:
                    correctly_predicted_adversarial_for_class_map[single_target.item()] += 1

            # To check the real adversarial status for the same predicted class
            if adversarial == 1:
                generated_adversarial_for_class[single_target.item()] += 1 

            num_of_items_in_class[single_target.item()] += 1

            ###################################################################################################################################
        
    for predicted, generated in zip(correctly_predicted_adversarial_for_class_map, generated_adversarial_for_class):
        correctly_predicted_adversarial_ratio_map.append((predicted/generated)*100)
    
    for predicted, generated in zip(correctly_predicted_adversarial_for_class_range, generated_adversarial_for_class): # Range
        correctly_predicted_adversarial_ratio_range.append((predicted/generated)*100)
    
    correctly_predicted_adversarial_ratio_map = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_map]

    correctly_predicted_adversarial_ratio_range = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_range] # Range

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
    parser = argparse.ArgumentParser(description="LenNet5 Network with MNIST Dataset")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help="Initial learning rate")
    parser.add_argument('--phase', default='train', help="train, test, profile, FGSM, sparsity-attack, sparsity-detect")
    parser.add_argument('--method', default='sparsity-map', help="profiling can be performed based on sparsity-map or sparsity-range")
    parser.add_argument('--weights', default=None, help="The path to the saved weights. Should be specified when testing")
    parser.add_argument('--dataset', default=None, help="The path to the train and test datasets")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon in each inner loop iteration")
    parser.add_argument('--constrained', action='store_true', help="To avoid clipping the generated purturbed data")
    parser.add_argument('--store_attack', action='store_true', help="To lead the algorithm to store the generated adversarials")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN architecture")
    parser.add_argument('--adversarial', action='store_true', help="To test the adversarial dataset in the test phase")
    parser.add_argument('--arch', default='cnvlutin', help="To specify the architecture running the clean/adversarial image: cnvlutin or dadiannao")
    
    args = parser.parse_args()
    print(f"{args}\n")

    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n A {device} is assigned for processing! \n")

    # MNIST dataset and dataloader declaration
    if args.phase == 'FGSM':
        test_dataset = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=True)
    elif args.phase == 'sparsity-attack':
        test_dataset = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.phase == 'sparsity-detect':
        # To load the adversarial_dataset from the offline file generated by sparsity-attack function (50% are polluted)
        test_dataset = torch.load(f"{args.dataset}/adversarial_dataset_constrained_10k.pt", map_location=device)
    elif args.phase == 'profile':
        train_dataset = mnist.MNIST(root=args.dataset, train=True, download=True, transform=transforms.ToTensor())
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
    elif args.adversarial:
        # To load the GAN/EA dataset from the offline file generated by GAN/EA algorithm
        # test_dataset = torch.load('./adversarial_data/Reconstruct_sparasity_constrained_L5000_R100_lr50.pt', map_location=device) # To examine the GAN testset
        test_dataset = torch.load('./adversarial_data/adversarial_dataset_constrained.pt', map_location=device) # To examine the sparsity attack
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)
    else:
        test_dataset  = mnist.MNIST(root=args.dataset, train=False, transform=transforms.ToTensor()) # Original Testset
        test_loader   = DataLoader(test_dataset, batch_size=args.batch_size)
        train_dataset = mnist.MNIST(root=args.dataset, train=True, download=True, transform=transforms.ToTensor())
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size)

    # Network Initialization
    model = LeNet5(args).to(device)
    print(f"\n{model}\n")
    
    if args.phase == 'train':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        loss_fn = CrossEntropyLoss()
        EPOCHS = args.epochs
        prev_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            for idx, (train_x, train_label) in enumerate(train_loader):
                optimizer.zero_grad()
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                preds = model(train_x.float(), False, False)
                loss = loss_fn(preds[0], train_label.long())
                loss.backward()
                optimizer.step()
    
            correct_num = 0
            sample_num = 0
            model.eval()
            for idx, (image, label) in enumerate(test_loader):
                image = image.to(device)
                label = label.to(device)
                preds = model(image.float().detach(), False, False)
                preds =torch.argmax(preds[0], dim=-1)
                current_correct_num = preds == label
                correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                sample_num += current_correct_num.shape[0]
            acc = correct_num / sample_num
            print('Accuracy in Epoch %3d: %1.4f' % (epoch+1, acc*100), flush=True)
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model.state_dict(), 'models/mnist_{:.4f}.pkl'.format(acc))
            if np.abs(acc - prev_acc) < 1e-4:
                break
            prev_acc = acc
        print("Training the model has been finished")

    elif args.phase == 'test':
        if args.weights is not None:
            # Load the pretrained model
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()

        correct_num = 0
        sample_num = 0
        
        total_dynamic_power_of_dataset = 0
        total_static_power_of_dataset = 0
        total_num_cycles_of_dataset = 0
        total_inference_delay = 0
        critical_path_delay = 20035 * 1e-12 # Critical path delay (in second) of CNVLUTIN = Multiplier(3732) + AdderTree(16179) + Relu(124)
        
        zeros_all_layers = [0] * 5
        sizes_all_layers = [0] * 5
        net_zeros  =  0
        net_sizes  =  0

        # Set the model in evaluation mode
        model.eval()
        # for (index, input) in enumerate(tqdm(test_loader, desc='Data Progress')): # To examine GAN dataset
        # for index, (image, label, adversarial) in enumerate(tqdm(test_dataset, desc='Data Progress')): # To examine sparsity attack dataset
        for index, (image, label) in enumerate(tqdm(test_loader, desc='Data Progress')): # To examine original dataset
            image = image.to(device)
            label = label.to(device)
            #image = input[0].to(device) # GAN
            #label = input[1].to(device) # GAN
            output = model(image, True, args.power)
                   
            # Sparsity calculations zone begin
            for i in range(5): zeros_all_layers[i] += output[1][i]
            for i in range(5): sizes_all_layers[i] += output[2][i]
            net_zeros += sum(output[1])
            net_sizes += sum(output[2])
            # Sparsity calculations zone end
            
            # Power consumption profiling zone start
            total_dynamic_power_of_dataset += output[4]
            total_static_power_of_dataset += output[5]
            total_num_cycles_of_dataset += output[6]
            total_inference_delay += output[7]
            # Power consumption profiling zone end
            
            preds = torch.argmax(output[0], dim=-1)
            current_correct_num = preds == label
            correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            sample_num += current_correct_num.shape[0]

        # To print accuracy statistics        
        acc = correct_num / sample_num
        print()
        print('Accuracy of Testing is (percent): %1.2f' % (acc*100))
        
        # To print sparsity rate statistics
        SR_L1  = (zeros_all_layers[0]/sizes_all_layers[0]) if sizes_all_layers[0] != 0 else 0
        SR_L2  = (zeros_all_layers[1]/sizes_all_layers[1]) if sizes_all_layers[1] != 0 else 0
        SR_L3  = (zeros_all_layers[2]/sizes_all_layers[2]) if sizes_all_layers[2] != 0 else 0
        SR_L4  = (zeros_all_layers[3]/sizes_all_layers[3]) if sizes_all_layers[3] != 0 else 0
        SR_L5  = (zeros_all_layers[4]/sizes_all_layers[4]) if sizes_all_layers[4] != 0 else 0
        SR_Net = (net_zeros/net_sizes) if net_sizes != 0 else 0
        print()
        print('Sparsity rate of L1 is: %1.5f'   % (SR_L1))
        print('Sparsity rate of L2 is: %1.5f'   % (SR_L2))
        print('Sparsity rate of L3 is: %1.5f'   % (SR_L3))
        print('Sparsity rate of L4 is: %1.5f'   % (SR_L4))
        print('Sparsity rate of L5 is: %1.5f'   % (SR_L5))
        print('Sparsity rate of Network: %1.5f' % (SR_Net))
        
        # To print power consumption statistics # power
        avg_dynamic_power_of_dataset = total_dynamic_power_of_dataset / sample_num
        avg_static_power_of_dataset = total_static_power_of_dataset / sample_num
        avg_dynamic_energy_of_dataset = avg_dynamic_power_of_dataset * (critical_path_delay) * total_num_cycles_of_dataset
        avg_static_energy_of_dataset = avg_static_power_of_dataset * (critical_path_delay) * total_num_cycles_of_dataset
        total_energy = avg_dynamic_energy_of_dataset + avg_static_energy_of_dataset
        total_delay = total_inference_delay / sample_num
        print()
        print('Average Dynamic Power: %1.9f (W)' % (avg_dynamic_power_of_dataset))
        print('Average Static Power: %1.9f (W)' % (avg_static_power_of_dataset))
        print('Average Dynamic Energy: %1.9f (J)' % (avg_dynamic_energy_of_dataset))
        print('Average Static Energy: %1.9f (J)' % (avg_static_energy_of_dataset))
        print('Total Energy: %1.9f (J)' % (total_energy))
        print('Average Inference Delay: %1.9f (Sec)' % (total_delay))
        
    elif args.phase == 'FGSM':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()

        # FGSM Initialization
        epsilons = [0, 0.007, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        # Set the model in evaluation mode
        model.eval()
        accuracies = []
        examples = []

        # Run test for each epsilon
        for eps in epsilons:
            acc, ex = FGSM_Test(model, device, test_loader, eps)
            accuracies.append(acc)
            examples.append(ex)

        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
    
    elif args.phase == 'sparsity-attack':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        num_classes  = len(test_dataset.classes)

        c_init = 0.5
        adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total = Sparsity_Attack_Generation(model, device, test_loader, num_classes, c_init, args)
        
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
        print(f"P1 = {index/(len(train_loader)-1)*100} % of training-set has been used for profiling.")

        if args.method == 'sparsity-map':
            # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_maps, "./adversarial_data/sparsity_maps.pt", pickle_protocol=4)
        
        if args.method == 'sparsity-range':
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

        # Load the sparsity-maps and sparsity-ranges from offline file (generated by profiling training set)
        offline_sparsity_maps = torch.load("./adversarial_data/sparsity_maps.pt")
        offline_sparsity_ranges = torch.load("./adversarial_data/sparsity_ranges.pth")
        
        num_classes  = len(offline_sparsity_maps) # or length of (offline_sparsity_ranges)

        # Prints the number of detected adversarials in each class
        Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes)

# Training:         python3 main.py --phase train
# Testing:          python3 main.py --phase test --power --arch cnvlutin --batch_size 10 --adversarial --dataset ./test --weights models/mnist_0.9876.pkl
# FGSM:             python3 main.py --phase FGSM --weights models/mnist_0.9876.pkl
# Profiling:        python3 main.py --phase profile --method sparsity-map/range --dataset train --weights models/mnist_0.9876.pkl
# Attack Execution: python3 main.py --phase sparsity-attack --eps 0.99 --eps_iter 0.04 --imax 200 --beta 31 --batch_size 10 --store-attack --constrained --dataset ./test --weights models/mnist_0.9876.pkl
# Attack Detection: python3 main.py --phase sparsity-detect --dataset adversarial_data --weights models/mnist_0.9876.pkl