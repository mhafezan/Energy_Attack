import ctypes
import math
import torch
import argparse
import sys
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# To load the shared library
lib = ctypes.CDLL('c_functions/lib_power_functions.so')

# To specify the argument and return types of the C function
lib.count_conv_multiplications.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_int,
                                           ctypes.POINTER(ctypes.c_double),
                                           ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.c_int))

lib.count_fc_multiplications.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.POINTER(ctypes.c_double),
                                         ctypes.POINTER(ctypes.c_int),
                                         ctypes.POINTER(ctypes.c_int))

lib.count_conv_multiplications.restype = None
lib.count_fc_multiplications.restype = None

# Custom Dataset to store the adversarial data
class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj, fc_in, args):
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
      
    def forward(self, x, SR=True, PC=True):

        zeros    = []
        sizes    = []
        eng_dnmc = []
        num_cycles = []
        
        dnmc_detail_b1, dnmc_detail_b21, dnmc_detail_b22, dnmc_detail_b31, dnmc_detail_b32, dnmc_detail_b33, dnmc_detail_b4 = ([] for _ in range(7))
        cycles_detail_b1, cycles_detail_b21, cycles_detail_b22, cycles_detail_b31, cycles_detail_b32, cycles_detail_b33, cycles_detail_b4  = ([] for _ in range(7))

        if PC:
            dnmc, cycles, dnmc_detail_b1, cycles_detail_b1 = conv_power(x, self.b1[0].weight, self.b1[0].stride, self.b1[0].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            b1_zeros, b1_sizes = sparsity_rate(x)
            zeros.append(b1_zeros)
            sizes.append(b1_sizes)
        s = self.b1[0](x) # conv2
        s = self.b1[1](s) # batch
        s = self.b1[2](s) # relu

        if PC:
            dnmc, cycles, dnmc_detail_b21, cycles_detail_b21 = conv_power(x, self.b2[0].weight, self.b2[0].stride, self.b2[0].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        # We don't need to compute SR(x) at this point, as we already computed it at b1 layer
        t = self.b2[0](x) # conv2
        t = self.b2[1](t) # batch
        t = self.b2[2](t) # relu

        if PC:
            dnmc, cycles, dnmc_detail_b22, cycles_detail_b22 = conv_power(t, self.b2[3].weight, self.b2[3].stride, self.b2[3].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            b2_zeros, b2_sizes = sparsity_rate(t)
            zeros.append(b2_zeros)
            sizes.append(b2_sizes)
        t = self.b2[3](t) # conv2
        t = self.b2[4](t) # batch
        t = self.b2[5](t) # relu

        if PC:
            dnmc, cycles, dnmc_detail_b31, cycles_detail_b31 = conv_power(x, self.b3[0].weight, self.b3[0].stride, self.b3[0].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        # We don't need to compute SR(x) and SM(x) at this point, as we already computed it at b1 layer
        u = self.b3[0](x) # conv2
        u = self.b3[1](u) # batch
        u = self.b3[2](u) # relu

        if PC:
            dnmc, cycles, dnmc_detail_b32, cycles_detail_b32 = conv_power(u, self.b3[3].weight, self.b3[3].stride, self.b3[3].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            b3_zeros, b3_sizes = sparsity_rate(u)
            zeros.append(b3_zeros)
            sizes.append(b3_sizes)
        u = self.b3[3](u) # conv2
        u = self.b3[4](u) # batch
        u = self.b3[5](u) # relu

        if PC:
            dnmc, cycles, dnmc_detail_b33, cycles_detail_b33 = conv_power(u, self.b3[6].weight, self.b3[6].stride, self.b3[6].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            b3_zeros, b3_sizes = sparsity_rate(u)
            zeros.append(b3_zeros)
            sizes.append(b3_sizes)
        u = self.b3[6](u) # conv2
        u = self.b3[7](u) # batch
        u = self.b3[8](u) # relu
     
        v = self.b4[0](x) # MaxPool2d

        if PC:
            dnmc, cycles, dnmc_detail_b4, cycles_detail_b4 = conv_power(v, self.b4[1].weight, self.b4[1].stride, self.b4[1].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            b4_zeros, b4_sizes = sparsity_rate(v)
            zeros.append(b4_zeros)
            sizes.append(b4_sizes)
        v = self.b4[1](v) # conv2
        v = self.b4[2](v) # batch
        v = self.b4[3](v) # relu
        
        eng_dnmc_detail = [a + b + c + d + e + f + g for a, b, c, d, e, f, g in zip(dnmc_detail_b1, dnmc_detail_b21, dnmc_detail_b22, dnmc_detail_b31, dnmc_detail_b32, dnmc_detail_b33, dnmc_detail_b4)]
        cycles_detail = [a + b + c + d + e + f + g for a, b, c, d, e, f, g in zip(cycles_detail_b1, cycles_detail_b21, cycles_detail_b22, cycles_detail_b31, cycles_detail_b32, cycles_detail_b33, cycles_detail_b4)]

        return torch.cat([s, t, u, v], dim=1), zeros, sizes, sum(eng_dnmc), sum(num_cycles), eng_dnmc_detail, cycles_detail
    
class GoogleNet(nn.Module):
    
    def __init__(self, num_class=100, args=None):
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
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, 16, args)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, 16, args)

        # In general, an Inception network is a network consisting of modules of the above type stacked upon each other, with occasional
        # max-pooling layers with stride 2 to halve the resolution of the grid
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, 8, args)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64, 8, args)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, 8, args)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, 8, args)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, 8, args)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, 4, args)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, 4, args)

        # Input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)
       
        # For computing sum of ones
        self.my_fc_pl_0 = nn.Linear(3*32*32,  1, bias=False)
        self.my_fc_pl_1 = nn.Linear(64*32*32, 1, bias=False)
        self.my_fc_pl_2 = nn.Linear(64*32*32, 1, bias=False)

        self.my_fc_12 = nn.Linear(12, 1, bias=False)

    def forward(self, x, SR=True, PC=True):
        # We break forward function into the minor layers: refer to https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/googlenet.py for the original function
        
        # To initialize the zeros and sizes with zero value
        L1_zeros, L2_zeros, L3_zeros = [0] * 3
        L4_L8_zeros, L9_L13_zeros, L14_L18_zeros, L19_L23_zeros, L24_L28_zeros, L29_L33_zeros, L34_L38_zeros, L39_L43_zeros, L44_L48_zeros = ([] for _ in range(9))
        L1_sizes, L2_sizes, L3_sizes = [0] * 3
        L4_L8_sizes, L9_L13_sizes, L14_L18_sizes, L19_L23_sizes, L24_L28_sizes, L29_L33_sizes, L34_L38_sizes, L39_L43_sizes, L44_L48_sizes = ([] for _ in range(9))
        
        L1_cycles_detail, L2_cycles_detail, L3_cycles_detail, L4_L8_cycles_detail, L9_L13_cycles_detail, L14_L18_cycles_detail, L19_L23_cycles_detail, L24_L28_cycles_detail, L29_L33_cycles_detail, L34_L38_cycles_detail, L39_L43_cycles_detail, L44_L48_cycles_detail = ([] for _ in range(12))
        L1_dnmc_detail, L2_dnmc_detail, L3_dnmc_detail, L4_L8_dnmc_detail, L9_L13_dnmc_detail, L14_L18_dnmc_detail, L19_L23_dnmc_detail, L24_L28_dnmc_detail, L29_L33_dnmc_detail, L34_L38_dnmc_detail, L39_L43_dnmc_detail, L44_L48_dnmc_detail = ([] for _ in range(12))
        
        eng_dnmc = []
        num_cycles = []
        
        if PC:
            dnmc, cycles, L1_dnmc_detail, L1_cycles_detail = conv_power(x, self.prelayer[0].weight, self.prelayer[0].stride, self.prelayer[0].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            L1_zeros, L1_sizes = sparsity_rate(x)
        x = self.prelayer[0](x) # conv2
        x = self.prelayer[1](x) # batch
        x = self.prelayer[2](x) # relu

        if PC:
            dnmc, cycles, L2_dnmc_detail, L2_cycles_detail = conv_power(x, self.prelayer[3].weight, self.prelayer[3].stride, self.prelayer[3].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            L2_zeros, L2_sizes = sparsity_rate(x)
        x = self.prelayer[3](x) # conv2
        x = self.prelayer[4](x) # batch
        x = self.prelayer[5](x) # relu

        if PC:
            dnmc, cycles, L3_dnmc_detail, L3_cycles_detail = conv_power(x, self.prelayer[6].weight, self.prelayer[6].stride, self.prelayer[6].padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            L3_zeros, L3_sizes = sparsity_rate(x)
        x = self.prelayer[6](x) # conv2
        x = self.prelayer[7](x) # batch
        x = self.prelayer[8](x) # relu

        x = self.maxpool(x)

        x, L4_L8_zeros, L4_L8_sizes, L4_L8_dnmc, L4_L8_cycles, L4_L8_dnmc_detail, L4_L8_cycles_detail = self.a3(x, SR, PC)
        
        x, L9_L13_zeros, L9_L13_sizes, L9_L13_dnmc, L9_L13_cycles, L9_L13_dnmc_detail, L9_L13_cycles_detail = self.b3(x, SR, PC)

        x = self.maxpool(x)

        x, L14_L18_zeros, L14_L18_sizes, L14_L18_dnmc, L14_L18_cycles, L14_L18_dnmc_detail, L14_L18_cycles_detail = self.a4(x, SR, PC)

        x, L19_L23_zeros, L19_L23_sizes, L19_L23_dnmc, L19_L23_cycles, L19_L23_dnmc_detail, L19_L23_cycles_detail = self.b4(x, SR, PC)

        x, L24_L28_zeros, L24_L28_sizes, L24_L28_dnmc, L24_L28_cycles, L24_L28_dnmc_detail, L24_L28_cycles_detail = self.c4(x, SR, PC)
        
        x, L29_L33_zeros, L29_L33_sizes, L29_L33_dnmc, L29_L33_cycles, L29_L33_dnmc_detail, L29_L33_cycles_detail = self.d4(x, SR, PC)

        x, L34_L38_zeros, L34_L38_sizes, L34_L38_dnmc, L34_L38_cycles, L34_L38_dnmc_detail, L34_L38_cycles_detail = self.e4(x, SR, PC)

        x = self.maxpool(x)

        x, L39_L43_zeros, L39_L43_sizes, L39_L43_dnmc, L39_L43_cycles, L39_L43_dnmc_detail, L39_L43_cycles_detail = self.a5(x, SR, PC)

        x, L44_L48_zeros, L44_L48_sizes, L44_L48_dnmc, L44_L48_cycles, L44_L48_dnmc_detail, L44_L48_cycles_detail = self.b5(x, SR, PC)

        x = self.avgpool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        
        zeros_list = [L1_zeros] + [L2_zeros] + [L3_zeros] + L4_L8_zeros + L9_L13_zeros + L14_L18_zeros + L19_L23_zeros + L24_L28_zeros + L29_L33_zeros + L34_L38_zeros + L39_L43_zeros + L44_L48_zeros
        sizes_list = [L1_sizes] + [L2_sizes] + [L3_sizes] + L4_L8_sizes + L9_L13_sizes + L14_L18_sizes + L19_L23_sizes + L24_L28_sizes + L29_L33_sizes + L34_L38_sizes + L39_L43_sizes + L44_L48_sizes
        
        pow_stat_total_inference, pow_stat_detail_inference = static_power()
        
        cycles_detail_inference = [a + b + c + d + e + f + g + h + i + j + k + l for a, b, c, d, e, f, g, h, i, j, k, l in zip(L1_cycles_detail, L2_cycles_detail, L3_cycles_detail, L4_L8_cycles_detail, L9_L13_cycles_detail, L14_L18_cycles_detail, L19_L23_cycles_detail, L24_L28_cycles_detail, L29_L33_cycles_detail, L34_L38_cycles_detail, L39_L43_cycles_detail, L44_L48_cycles_detail)]
        
        eng_dnmc_detail_inference = [a + b + c + d + e + f + g + h + i + j + k + l for a, b, c, d, e, f, g, h, i, j, k, l in zip(L1_dnmc_detail, L2_dnmc_detail, L3_dnmc_detail, L4_L8_dnmc_detail, L9_L13_dnmc_detail, L14_L18_dnmc_detail, L19_L23_dnmc_detail, L24_L28_dnmc_detail, L29_L33_dnmc_detail, L34_L38_dnmc_detail, L39_L43_dnmc_detail, L44_L48_dnmc_detail)]
        
        eng_stat_detail_inference = [a * b for a, b in zip(pow_stat_detail_inference, cycles_detail_inference)]
        
        eng_dnmc_total_inference = sum(eng_dnmc) + L4_L8_dnmc + L9_L13_dnmc + L14_L18_dnmc + L19_L23_dnmc + L24_L28_dnmc + L29_L33_dnmc + L34_L38_dnmc + L39_L43_dnmc + L44_L48_dnmc
        
        eng_stat_total_inference = pow_stat_total_inference * (sum(num_cycles) + L4_L8_cycles + L9_L13_cycles + L14_L18_cycles + L19_L23_cycles + L24_L28_cycles + L29_L33_cycles + L34_L38_cycles + L39_L43_cycles + L44_L48_cycles)
        
        return x, zeros_list, sizes_list, eng_dnmc_total_inference, eng_stat_total_inference, eng_dnmc_detail_inference, eng_stat_detail_inference

def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

def static_power():

    pow_static_breakdown = [None] * 10

    pow_stat_multipliers = 0.061512900608 # 16*256 multipliers are accessed within a node in each cycle: (16*256)*0.000015017798 (W)
    pow_stat_adders = 0.032082976 # Number of accessed adders in each cycle: (16*16)*0.000125324125 (W)
    pow_stat_encoder = 0.00101210944 # Number of accessed encoders in each cycle: 16*63256.84e-9 (W)
    pow_stat_dispatcher = 0.000511910231 # (W)
    pow_stat_relu = 0.00006037248 # 256*235.83e-9: 16 non-zeros are dispatched, and always 256 outputs are generated simultaneously, and are activated by 256 ReLUs (W)
    pow_stat_nbin = 0.182776576 # 256*(0.713971) (W)
    pow_stat_offset = 0.182776576 # 256*(0.713971) (W)
    pow_stat_SB = 0.00131820288 # 256*(0.00514923) (W)
    pow_stat_nbout = 0.04516752 # 16*(2.82297) (W)
    pow_stat_nm = 0.0132751 # (13.2751) (W)
    pow_static = pow_stat_multipliers + pow_stat_adders + pow_stat_relu + pow_stat_encoder + pow_stat_dispatcher + pow_stat_nbin + pow_stat_offset + pow_stat_SB + pow_stat_nbout + pow_stat_nm
    pow_static_breakdown = [pow_stat_multipliers, pow_stat_adders, pow_stat_relu, pow_stat_encoder, pow_stat_dispatcher, pow_stat_nbin, pow_stat_offset, pow_stat_SB, pow_stat_nbout, pow_stat_nm]

    return pow_static, pow_static_breakdown

def conv_power (input, weight, stride, padding, architecture):

    padding = (padding[1], padding[1], padding[0], padding[0])
    input = F.pad(input, padding, mode='constant', value=0)
    
    cycles_total = None
    eng_dynamic_total = None
    eng_dynamic_breakdown = [None] * 10
    
    num_input_elements = input.numel()
    num_non_zero_inputs = torch.count_nonzero(input).item()
    num_output_elements = input.size(0) * weight.size(0) * ((input.size(2) - weight.size(2) + 1)//stride[0]) * ((input.size(3) - weight.size(3) + 1)//stride[1])
            
    """
    num_all_mul = 0
    num_non_zero_mul = 0
    """
    """
    percent_non_zero_inputs = num_non_zero_inputs/num_input_elements
    num_all_mul = input.size(0) * weight.size(0) * ((input.size(2) - weight.size(2) + 1)//stride[0]) * ((input.size(3) - weight.size(3) + 1)//stride[1]) * input.size(1) * weight.size(2) * weight.size(3)
    num_non_zero_mul = percent_non_zero_inputs * num_all_mul
    """
    """
    # Iterate over each element according to the convolution functionality
    for b in range(input.size(0)): # Batch dimension
        for c_out in range(weight.size(0)): # Output channel dimension (to iterate over different kernels)
            for h_out in range(0, input.size(2) - weight.size(2) + 1, stride[0]): # Output Height dimension
                for w_out in range(0, input.size(3) - weight.size(3) + 1, stride[1]): # Output Width dimension
                    for c_in in range(input.size(1)): # Input/Kernel channel dimension (Kernel depth)
                        for h_k in range(weight.size(2)): # Kernel height dimension
                            for w_k in range(weight.size(3)): # Kernel width dimension
                                num_all_mul += 1
                                input_operand = input[b, c_in, h_out + h_k, w_out + w_k]
                                if input_operand.item() != 0:
                                    num_non_zero_mul += 1
    """

    # To ensure the input tensor is on the CPU size and flatten the 4D input tensor into 1D
    input_flattened = input.detach().cpu().flatten().numpy().astype(np.float64)
    
    # Allocate memory for the output variables
    num_all_mul = ctypes.c_int()
    num_non_zero_mul = ctypes.c_int()
    
    # To call the C function
    lib.count_conv_multiplications(input.size(0), input.size(1), input.size(2),input.size(3),
                                   weight.size(0), weight.size(2), weight.size(3),
                                   stride[0], stride[1],
                                   input_flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   ctypes.byref(num_all_mul),
                                   ctypes.byref(num_non_zero_mul))
    
    num_all_mul = num_all_mul.value
    num_non_zero_mul = num_non_zero_mul.value
    
    if (architecture == "dadiannao"):
        num_non_zero_mul = num_all_mul
    
    eng_dnmc_multipliers_per_access = 414521.521e-9 * 3732e-12 # Energy per access to a Multiplier (J)
    eng_dnmc_multipliers = num_non_zero_mul * eng_dnmc_multipliers_per_access # Number of accesses to all multipliers equals to the non-zero multiplications (J)
    cycles_multipliers = math.ceil(num_non_zero_mul/4096) # Required cycles to perform all non-zero multiplications in parallel: num_non_zero_mul/(16*256)
            
    eng_dnmc_adders_per_access = 6130416.161e-9 * 16179e-12 # Energy per access to a Adder Tree (J)
    eng_dnmc_adders = (num_non_zero_mul/16) * eng_dnmc_adders_per_access # Number of accesses to all adders: (num_non_zero_mul/16) (J)
    cycles_adders = 0 # It has overlap with NFU cycles
            
    eng_dnmc_encoder_per_access = 581244.983e-9 * 1736e-12 # Energy per access to a Encoder (J)
    eng_dnmc_encoder = (num_output_elements/16) * eng_dnmc_encoder_per_access # Number of accesses to all encoders within a node: (num_output_elements/256)*16 (J)
    cycles_encoder = math.ceil(num_output_elements/16) # Required cycles to encode all outputs by 16 parallel encoders in 16 cycles: (num_output_elements/256)*16
            
    # The number of accesses to Dispatcher is equal to the number of bricks in NM (or Input Tensor), as each brick in each lane should be proccessed
    # independently by one of 16 parallel lanes in the Dispatcher. Each Brick Buffer (BB) Entry in the Dispatcher is connected to corresponding NM Bank
    # via a 256-bit BUS (16-neuron-wide) and is able to read 256 bits (i.e., one brick) within 4 cycles (NM delay).
    eng_dnmc_dispatcher_per_access = 7665235.699e-9 * 1825e-12 # Energy per access to the Dispatcher (J)
    eng_dnmc_dispatcher = (num_input_elements/16) * eng_dnmc_dispatcher_per_access # Number of accesses to Dispatcher: (num_input_elements/16) (J)
    cycles_dispatcher = math.ceil(num_input_elements/16) if architecture == "dadiannao" else math.ceil(num_non_zero_inputs/16)
    # If 16 non-zero inputs are detected simultaneously in all Dispatcher lanes over a cycle, one shift to the output is done, otherwise Dispatcher is clock-gated.
    # Accordingly, we devided the num_non_zero_inputs by 16 (Paper: Dispatcher broadcasts non-zero neurons, a single-neuron from each BB entry at a time, for a total
    # of 16 neurons, one per BB entry and thus per neuron lane each cycle).
            
    eng_dnmc_relu_per_access = 1865.64e-9 * 124e-12 # Energy per access to a ReLU (J)
    eng_dnmc_relu = num_output_elements * eng_dnmc_relu_per_access # ReLU component is activated once a single output is generated (J)
    cycles_relu = 0 # It has overlap with NFU cycles
            
    eng_dnmc_nbin_per_access = 0.000874423e-9 + 0.00202075e-9 # Dynamic (read energy + write energy) per access to NBin; The same number of accesses happens for read and write
    eng_dnmc_nbin = (num_non_zero_inputs * 16) * eng_dnmc_nbin_per_access if architecture == "cnvlutin" else (num_input_elements * 16) * eng_dnmc_nbin_per_access # ((num_non_zero_inputs/16)*256)*(Read+Write Energy) (J)
    cycles_nbin = math.ceil(num_non_zero_inputs/16) if architecture == "cnvlutin" else math.ceil(num_input_elements/16) # Cycles to Write into 256 parallel Nbins (Read Cycles are covered by NFU cycles)
            
    eng_dnmc_offset_per_access = 0.000874423e-9 # Dynamic read energy per access to Offset (J)
    eng_dnmc_offset = (num_non_zero_inputs * 16) * eng_dnmc_offset_per_access if architecture == "cnvlutin" else (num_input_elements * 16) * eng_dnmc_offset_per_access
    cycles_offset = 0 # It has overlap with nbin cycles (Paper: Every cycle, each subunit fetches a single (neuron, offset) pair from NBin, uses the
    # offset to index the corresponding entry from its SBin to fetch 16 synapses and produces 16 products, one per filter)
            
    eng_dnmc_SB_per_access = 0.00317674e-9 # Dynamic read energy per access to SB (J)
    eng_dnmc_SB = (num_non_zero_inputs * 16) * eng_dnmc_SB_per_access if architecture == "cnvlutin" else (num_input_elements * 16) * eng_dnmc_SB_per_access # 16 entries of one SB are accessed in 1 cycle: ((num_non_zero_inputs/16)*256) (J)
    cycles_SB = 0 # It has overlap with nbin cycles (Paper: we decided to clock the NFU at the same frequency as the eDRAM available in 28nm technology, Thus SB cycles is covered by one NFU cycle)
            
    eng_dnmc_nbout = (num_output_elements/16) * (0.00300943e-9) + (num_output_elements/16 + num_non_zero_mul/256) * (0.00183095e-9) # (Read+Write Energy)*Accesses (J)
    cycles_nbout = 0 # Read Cycles are covered by encoder cycles and Write Cycles are covered by NFU Cycles.
            
    eng_dnmc_nm = (num_input_elements * 1.6507625e-12) + (num_output_elements * 40.45925e-12) # (num_input_elements/16)*(0.0264122e-9)+(num_output_elements/16)*(0.647348e-9) (J)
    cycles_nm = math.ceil(num_input_elements/16) # X 1 cycle: Number of cycles required to read all bricks from NM by Dispatcher (Write cycles, i.e., num_output_elements/16 overlap
    # with Pipeline Cycles). The NM latency for DaDianNao architecture is 10-cycles, where NFU and eDRAM are clocked at 606 MHz in 28nm technology. As NM cycle time (from CACTI)
    # in 45nm is 7.32542 nS, maintaining 606 MHz frequency for the NFU, we can compute the NM latency for 45nm as follows:
    # (7.32542/2)=3.66271 ~ 4 cycles. However, our designed NFU critical path delay is 20035 ps, thus, (7.32542/20035)=0.365 ~ 1 cycle
            
    eng_dynamic_total = eng_dnmc_multipliers + eng_dnmc_adders + eng_dnmc_relu + eng_dnmc_encoder + eng_dnmc_dispatcher + eng_dnmc_nbin + eng_dnmc_offset + eng_dnmc_SB + eng_dnmc_nbout + eng_dnmc_nm
    eng_dynamic_breakdown = [eng_dnmc_multipliers, eng_dnmc_adders, eng_dnmc_relu, eng_dnmc_encoder, eng_dnmc_dispatcher, eng_dnmc_nbin, eng_dnmc_offset, eng_dnmc_SB, eng_dnmc_nbout, eng_dnmc_nm]
    
    cycles_total = cycles_multipliers + cycles_adders + cycles_relu + cycles_encoder + cycles_dispatcher + cycles_nbin + cycles_offset + cycles_SB + cycles_nbout + cycles_nm
    # ORIGINAL ORDER: cycles_multipliers, cycles_adders, cycles_relu, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_offset, cycles_SB, cycles_nbout, cycles_nm
    # For the num_cycles that overlap with other component's cycles, we reflect them out as below to correctly compute the static energy of each component using eng_static_breakdown
    cycles_breakdown = [cycles_multipliers, cycles_multipliers, cycles_multipliers, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_nbin, cycles_nbin, cycles_encoder, cycles_nm]
    
    return eng_dynamic_total, cycles_total, eng_dynamic_breakdown, cycles_breakdown

def fc_power (input, weight, architecture):

    eng_dynamic_total = None
    cycles_total = None
    eng_dynamic_breakdown = [None] * 10
    
    num_input_elements = input.numel()
    num_non_zero_inputs = torch.count_nonzero(input).item()
    num_output_elements = input.size(0) * weight.size(0)
    
    """
    num_all_mul = 0
    num_non_zero_mul = 0
    """
    """
    percent_non_zero_inputs = num_non_zero_inputs/num_input_elements
    num_all_mul = input.size(0) * weight.size(0) * input.size(1)
    num_non_zero_mul = percent_non_zero_inputs * num_all_mul
    """
    """
    # Iterate over each element of the input tensor and the weight tensor to compute the number of non-zero multiplications
    for i in range(input.size(0)): # Iterate over batch dimension
        for j in range(weight.size(0)): # Iterate over output dimension
            for k in range(input.size(1)): # Iterate over the identical-size dimension
                num_all_mul += 1
                if input[i, k].item() != 0:
                    num_non_zero_mul += 1
    """
    # To ensure the input tensor is on the CPU size and flatten the input tensor into 1D
    input_flattened = input.detach().cpu().flatten().numpy().astype(np.float64)
    
    # Allocate memory for the output variables
    num_all_mul = ctypes.c_int()
    num_non_zero_mul = ctypes.c_int()
    
    # To call the C function
    lib.count_fc_multiplications(input.size(0), input.size(1), weight.size(0),
                                   input_flattened.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                   ctypes.byref(num_all_mul),
                                   ctypes.byref(num_non_zero_mul))
    
    num_all_mul = num_all_mul.value
    num_non_zero_mul = num_non_zero_mul.value

    if (architecture == "dadiannao"):
        num_non_zero_mul = num_all_mul
            
    eng_dnmc_multipliers_per_access = 414521.521e-9 * 3732e-12 # Energy per access to a Multiplier (J)
    eng_dnmc_multipliers = num_non_zero_mul * eng_dnmc_multipliers_per_access # Number of accesses to all multipliers equals to the non-zero multiplications (J)
    cycles_multipliers = math.ceil(num_non_zero_mul/4096) # Required cycles to perform all non-zero multiplications in parallel: num_non_zero_mul/(16*256)
            
    eng_dnmc_adders_per_access = 6130416.161e-9 * 16179e-12 # Energy per access to a Adder Tree (J)
    eng_dnmc_adders = (num_non_zero_mul/16) * eng_dnmc_adders_per_access # Number of accesses to all adders: (num_non_zero_mul/16) (J)
    cycles_adders = 0 # It has overlap with NFU cycles
            
    eng_dnmc_encoder_per_access = 581244.983e-9 * 1736e-12 # Energy per access to a Encoder (J)
    eng_dnmc_encoder = (num_output_elements/16) * eng_dnmc_encoder_per_access # Number of accesses to all encoders within a node: (num_output_elements/256)*16 (J)
    cycles_encoder = math.ceil(num_output_elements/16) # Required cycles to encode all outputs by 16 parallel encoders in 16 cycles: (num_output_elements/256)*16
            
    # The number of accesses to Dispatcher is equal to the number of bricks in NM (or Input Tensor), as each brick in each lane should be proccessed
    # independently by one of 16 parallel lanes in the Dispatcher. Each Brick Buffer (BB) Entry in the Dispatcher is connected to corresponding NM Bank
    # via a 256-bit BUS (16-neuron-wide) and is able to read 256 bits (i.e., one brick) within 4 cycles (NM delay).
    eng_dnmc_dispatcher_per_access = 7665235.699e-9 * 1825e-12 # Energy per access to the Dispatcher (J)
    eng_dnmc_dispatcher = (num_input_elements/16) * eng_dnmc_dispatcher_per_access # Number of accesses to Dispatcher: (num_input_elements/16) (J)
    cycles_dispatcher = math.ceil(num_input_elements/16) if architecture == "dadiannao" else math.ceil(num_non_zero_inputs/16)
    # If 16 non-zero inputs are detected simultaneously in all Dispatcher lanes over a cycle, one shift to the output is done, otherwise Dispatcher is clock-gated.
    # Accordingly, we devided the num_non_zero_inputs by 16 (Paper: Dispatcher broadcasts non-zero neurons, a single-neuron from each BB entry at a time, for a total
    # of 16 neurons, one per BB entry and thus per neuron lane each cycle).
            
    eng_dnmc_relu_per_access = 1865.64e-9 * 124e-12 # Energy per access to a ReLU (J)
    eng_dnmc_relu = num_output_elements * eng_dnmc_relu_per_access # ReLU component is activated once a single output is generated (J)
    cycles_relu = 0 # It has overlap with NFU cycles
            
    eng_dnmc_nbin_per_access = 0.000874423e-9 + 0.00202075e-9 # Dynamic (read energy + write energy) per access to NBin; The same number of accesses happens for read and write
    eng_dnmc_nbin = (num_non_zero_inputs * 16) * eng_dnmc_nbin_per_access if architecture == "cnvlutin" else (num_input_elements * 16) * eng_dnmc_nbin_per_access # ((num_non_zero_inputs/16)*256)*(Read+Write Energy) (J)
    cycles_nbin = math.ceil(num_non_zero_inputs/16) if architecture == "cnvlutin" else math.ceil(num_input_elements/16) # Cycles to Write into 256 parallel Nbins (Read Cycles are covered by NFU cycles)
            
    eng_dnmc_offset_per_access = 0.000874423e-9 # Dynamic read energy per access to Offset (J)
    eng_dnmc_offset = (num_non_zero_inputs * 16) * eng_dnmc_offset_per_access if architecture == "cnvlutin" else (num_input_elements * 16) * eng_dnmc_offset_per_access
    cycles_offset = 0 # It has overlap with nbin cycles (Paper: Every cycle, each subunit fetches a single (neuron, offset) pair from NBin, uses the
    # offset to index the corresponding entry from its SBin to fetch 16 synapses and produces 16 products, one per filter)
            
    eng_dnmc_SB_per_access = 0.00317674e-9 # Dynamic read energy per access to SB (J)
    eng_dnmc_SB = (num_non_zero_inputs * 16) * eng_dnmc_SB_per_access if architecture == "cnvlutin" else (num_input_elements * 16) * eng_dnmc_SB_per_access # 16 entries of one SB are accessed in 1 cycle: ((num_non_zero_inputs/16)*256) (J)
    cycles_SB = 0 # It has overlap with nbin cycles (Paper: we decided to clock the NFU at the same frequency as the eDRAM available in 28nm technology, Thus SB cycles is covered by one NFU cycle)
            
    eng_dnmc_nbout = (num_output_elements/16) * (0.00300943e-9) + (num_output_elements/16 + num_non_zero_mul/256) * (0.00183095e-9) # (Read+Write Energy)*Accesses (J)
    cycles_nbout = 0 # Read Cycles are covered by encoder cycles and Write Cycles are covered by NFU Cycles.
            
    eng_dnmc_nm = (num_input_elements * 1.6507625e-12) + (num_output_elements * 40.45925e-12) # (num_input_elements/16)*(0.0264122e-9)+(num_output_elements/16)*(0.647348e-9) (J)
    cycles_nm = math.ceil(num_input_elements/16) # X 1 cycle: Number of cycles required to read all bricks from NM by Dispatcher (Write cycles, i.e., num_output_elements/16 overlap
    # with Pipeline Cycles). The NM latency for DaDianNao architecture is 10-cycles, where NFU and eDRAM are clocked at 606 MHz in 28nm technology. As NM cycle time (from CACTI)
    # in 45nm is 7.32542 nS, maintaining 606 MHz frequency for the NFU, we can compute the NM latency for 45nm as follows:
    # (7.32542/2)=3.66271 ~ 4 cycles. However, our designed NFU critical path delay is 20035 ps, thus, (7.32542/20035)=0.365 ~ 1 cycle
            
    eng_dynamic_total = eng_dnmc_multipliers + eng_dnmc_adders + eng_dnmc_relu + eng_dnmc_encoder + eng_dnmc_dispatcher + eng_dnmc_nbin + eng_dnmc_offset + eng_dnmc_SB + eng_dnmc_nbout + eng_dnmc_nm
    eng_dynamic_breakdown = [eng_dnmc_multipliers, eng_dnmc_adders, eng_dnmc_relu, eng_dnmc_encoder, eng_dnmc_dispatcher, eng_dnmc_nbin, eng_dnmc_offset, eng_dnmc_SB, eng_dnmc_nbout, eng_dnmc_nm]
    
    cycles_total = cycles_multipliers + cycles_adders + cycles_relu + cycles_encoder + cycles_dispatcher + cycles_nbin + cycles_offset + cycles_SB + cycles_nbout + cycles_nm
    # ORIGINAL ORDER: cycles_multipliers, cycles_adders, cycles_relu, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_offset, cycles_SB, cycles_nbout, cycles_nm
    # For the num_cycles that overlap with other component's cycles, we reflect them out as below to correctly compute the static energy of each component using eng_static_breakdown
    cycles_breakdown = [cycles_multipliers, cycles_multipliers, cycles_multipliers, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_nbin, cycles_nbin, cycles_encoder, cycles_nm]
    
    return eng_dynamic_total, cycles_total, eng_dynamic_breakdown, cycles_breakdown

if __name__ == '__main__':
    
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="GoogleNet Network with CIFAR100 Dataset")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weights', default="../2_copy_weight/googlenet_cifar100_fc_one_out.pkl", help="The path to the pre-trained weights")
    parser.add_argument('--dataset', default="../cifar100_dataset", help="The path to the train or test datasets")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN or DaDianNao architecture")
    parser.add_argument('--arch', default='cnvlutin', help="To specify the architecture running the clean/adversarial images: cnvlutin or dadiannao")
    parser.add_argument('--adversarial', action='store_true', help="To test the adversarial dataset instead of original dataset")
    parser.add_argument('--im_index_first', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=100, type=int, help="The last index of the dataset")
    parser.add_argument('--adv_images', default=None, help="The path to the adversarial images generated by energy attack algorithm")
    args = parser.parse_args()
    print(f"\n{args}\n")
    
    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{device} assigned for processing!\n")

    # CIFAR100 dataset and dataloader declaration
    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    if args.adversarial:
        # To load the GAN/EA dataset from the offline file generated by GAN/EA algorithm
        test_dataset = torch.load(args.adv_images, map_location=torch.device('cpu')) # To examine the sparsity attack
        test_dataset_sub = torch.utils.data.Subset(test_dataset, list(range(args.im_index_first, args.im_index_last)))                
        test_loader = DataLoader(test_dataset_sub, batch_size=args.batch_size, shuffle=False)
    else:             
        # To load original testset
        test_dataset = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=TRANSFORM)
        test_dataset_sub = torch.utils.data.Subset(test_dataset, list(range(args.im_index_first, args.im_index_last)))
        test_loader = DataLoader(test_dataset_sub, batch_size=args.batch_size, shuffle=False)
        
    # Network Initialization
    model = GoogleNet(args=args).to(device)
    
    # To load the pretrained model
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()

    correct_num = 0
    sample_num = 0
    num_layers = 48
    
    total_dynamic_energy_dataset = 0
    total_static_energy_dataset = 0
    eng_dnmc_detail_dataset = [0] * 10
    eng_stat_detail_dataset = [0] * 10
    
    # Read from (NBin+Offset+SB) + Multiplication + Add by AdderTree + Activating by ReLU + Write into NBout
    critical_path_delay = (0.201776e-9 + 0.201776e-9 + 0.861771e-9) + 3732e-12 + 16179e-12 + 124e-12 + (0.261523e-9)
        
    zeros_all_layers = [0] * num_layers
    sizes_all_layers = [0] * num_layers
    net_zeros  =  0
    net_sizes  =  0

    # Set the model in evaluation mode
    model.eval()
    
    for index, (image, label) in enumerate(tqdm(test_loader, desc='Data Progress')):
      
        image = image.to(device)
        label = label.to(device)

        output = model(image, True, args.power)
                   
        # Sparsity calculations
        for i in range(num_layers): zeros_all_layers[i] += output[1][i]
        for i in range(num_layers): sizes_all_layers[i] += output[2][i]
        net_zeros += sum(output[1])
        net_sizes += sum(output[2])
            
        # Power consumption profiling
        total_dynamic_energy_dataset += output[3]
        total_static_energy_dataset += output[4]
        eng_dnmc_detail_dataset = [a + b for a, b in zip(eng_dnmc_detail_dataset, output[5])]
        eng_stat_detail_dataset = [a + b for a, b in zip(eng_stat_detail_dataset, output[6])]

        # Prediction and Accuracy Measurement
        _, preds = torch.max(output[0].data, 1)
        sample_num += label.size(0)
        correct_num += (preds == label).sum().item()
        
    # To print accuracy statistics        
    acc = correct_num / sample_num
    print()
    print('Accuracy of Testing is (percent): %1.2f' % (acc*100), flush=True)
        
    # To print sparsity rate statistics
    SR = []
    for i in range(num_layers):
        SR.append((zeros_all_layers[i] / sizes_all_layers[i]) if sizes_all_layers[i] != 0 else 0)

    SR_Net = (net_zeros/net_sizes) if net_sizes != 0 else 0

    print()
    for i in range(num_layers):
        print(f"Sparsity rate of L{i+1} is: {SR[i]:.5f}")
    print('Sparsity rate of Network: %1.5f' % (SR_Net))
        
    # To print energy consumption statistics
    avg_dynamic_energy_dataset = total_dynamic_energy_dataset / sample_num
    avg_static_energy_dataset = (total_static_energy_dataset / sample_num) * critical_path_delay
    total_energy = avg_dynamic_energy_dataset + avg_static_energy_dataset
    eng_stat_detail_dataset = [x * critical_path_delay for x in eng_stat_detail_dataset]
    eng_total_detail_dataset = [a + b for a, b in zip(eng_dnmc_detail_dataset, eng_stat_detail_dataset)]

    if args.power:
        print()
        print('Average Dynamic Energy of Dataset: %1.9f (J)' % (avg_dynamic_energy_dataset))
        print('Average Static Energy of Dataset: %1.9f (J)' % (avg_static_energy_dataset))
        print('Total Energy of Dataset: %1.9f (J)' % (total_energy))
        print()
        print('########## Dynamic Energy Breakdown ##########')
        print()
        print('Multiplier (J): %1.20f' % (eng_dnmc_detail_dataset[0]))
        print('AdderTree  (J): %1.20f' % (eng_dnmc_detail_dataset[1]))
        print('ReLu       (J): %1.20f' % (eng_dnmc_detail_dataset[2]))
        print('Encoder    (J): %1.20f' % (eng_dnmc_detail_dataset[3]))
        print('Dispatcher (J): %1.20f' % (eng_dnmc_detail_dataset[4]))
        print('NBin       (J): %1.20f' % (eng_dnmc_detail_dataset[5]))
        print('Offset     (J): %1.20f' % (eng_dnmc_detail_dataset[6]))
        print('SB         (J): %1.20f' % (eng_dnmc_detail_dataset[7]))
        print('NBout      (J): %1.20f' % (eng_dnmc_detail_dataset[8]))
        print('NM         (J): %1.20f' % (eng_dnmc_detail_dataset[9]))
    
        total_eng_dnmc = sum(eng_dnmc_detail_dataset)
        print('Total      (J): %1.20f' % (total_eng_dnmc))
    
        print()
        print('########## Static Energy Breakdown ###########')
        print()
        print('Multiplier (J): %1.20f' % (eng_stat_detail_dataset[0]))
        print('AdderTree  (J): %1.20f' % (eng_stat_detail_dataset[1]))
        print('ReLu       (J): %1.20f' % (eng_stat_detail_dataset[2]))
        print('Encoder    (J): %1.20f' % (eng_stat_detail_dataset[3]))
        print('Dispatcher (J): %1.20f' % (eng_stat_detail_dataset[4]))
        print('NBin       (J): %1.20f' % (eng_stat_detail_dataset[5]))
        print('Offset     (J): %1.20f' % (eng_stat_detail_dataset[6]))
        print('SB         (J): %1.20f' % (eng_stat_detail_dataset[7]))
        print('NBout      (J): %1.20f' % (eng_stat_detail_dataset[8]))
        print('NM         (J): %1.20f' % (eng_stat_detail_dataset[9]))
    
        total_eng_stat = sum(eng_stat_detail_dataset)
        print('Total      (J): %1.20f' % (total_eng_stat))
    
        print()
        print('########## Total Energy Breakdown ############')
        print()
        print('Multiplier (J): %1.20f' % (eng_total_detail_dataset[0]))
        print('AdderTree  (J): %1.20f' % (eng_total_detail_dataset[1]))
        print('ReLu       (J): %1.20f' % (eng_total_detail_dataset[2]))
        print('Encoder    (J): %1.20f' % (eng_total_detail_dataset[3]))
        print('Dispatcher (J): %1.20f' % (eng_total_detail_dataset[4]))
        print('NBin       (J): %1.20f' % (eng_total_detail_dataset[5]))
        print('Offset     (J): %1.20f' % (eng_total_detail_dataset[6]))
        print('SB         (J): %1.20f' % (eng_total_detail_dataset[7]))
        print('NBout      (J): %1.20f' % (eng_total_detail_dataset[8]))
        print('NM         (J): %1.20f' % (eng_total_detail_dataset[9]))
    
        total_eng = sum(eng_total_detail_dataset)
        print('Total      (J): %1.20f' % (total_eng))    
        print()
    else:
        print()
        print("No data is available for Energy Consumption. Use --power as arguments to print power results")

    sys.exit(0)

# Arguments: python3 cifar100_power.py --power --arch cnvlutin --batch_size 1 --im_index_last 40 --dataset ../cifar100_dataset --adversarial --adv_images ../6_adv_attack_detect/adversarial_data/adv_constrained.pt --weights ../2_copy_weight/googlenet_cifar100_fc_one_out.pkl