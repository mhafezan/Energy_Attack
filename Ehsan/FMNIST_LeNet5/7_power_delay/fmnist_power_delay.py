import ctypes
import math
import torch
import argparse
import sys
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from tqdm import tqdm

# To load the shared library
lib = ctypes.CDLL('c_functions/lib_power_functions.so')

# To specify the argument and return types of the C functions
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
    
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, args=None):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
        # To compute sum of ones
        self.my_fc1 = nn.Linear(1*32*32, 1, bias=False)
        self.my_fc2 = nn.Linear(6*14*14, 1, bias=False)
        self.my_fc3 = nn.Linear(16*5*5, 1, bias=False)
        self.my_fc4 = nn.Linear(120, 1, bias=False)
        self.my_fc5 = nn.Linear(4, 1, bias=False)
    
    def forward(self, x, SR=True, PC=False):
        
        L1_zeros, L2_zeros, L3_zeros, L4_zeros = [0] * 4
        L1_sizes, L2_sizes, L3_sizes, L4_sizes = [0] * 4
        
        L1_cycles_detail, L2_cycles_detail, L3_cycles_detail, L4_cycles_detail = ([] for _ in range(4))
        L1_dnmc_detail, L2_dnmc_detail, L3_dnmc_detail, L4_dnmc_detail = ([] for _ in range(4))
        
        eng_dnmc = []
        num_cycles = []
        
        if PC:
            dnmc, cycles, L1_dnmc_detail, L1_cycles_detail = conv_power(x, self.conv1.weight, self.conv1.stride, self.conv1.padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            L1_zeros, L1_sizes = sparsity_rate(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        if PC:
            dnmc, cycles, L2_dnmc_detail, L2_cycles_detail = conv_power(x, self.conv2.weight, self.conv2.stride, self.conv2.padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            L2_zeros, L2_sizes = sparsity_rate(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        if PC:
            dnmc, cycles, L3_dnmc_detail, L3_cycles_detail = conv_power(x, self.conv3.weight, self.conv3.stride, self.conv3.padding, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            L3_zeros, L3_sizes = sparsity_rate(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(-1, 120)
        
        if PC:
            dnmc, cycles, L4_dnmc_detail, L4_cycles_detail = fc_power(x, self.fc1.weight, args.arch)
            eng_dnmc.append(dnmc)
            num_cycles.append(cycles)
        if SR:
            L4_zeros, L4_sizes = sparsity_rate(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        zeros_list = [L1_zeros, L2_zeros, L3_zeros, L4_zeros]
        sizes_list = [L1_sizes, L2_sizes, L3_sizes, L4_sizes]

        pow_stat_total_inference, pow_stat_detail_inference = static_power()
        
        cycles_detail_inference = [a+b+c+d for a, b, c, d in zip(L1_cycles_detail, L2_cycles_detail, L3_cycles_detail, L4_cycles_detail)]
        
        eng_dnmc_detail_inference = [a+b+c+d for a, b, c, d in zip(L1_dnmc_detail, L2_dnmc_detail, L3_dnmc_detail, L4_dnmc_detail)]
        
        eng_stat_detail_inference = [a*b for a, b in zip(pow_stat_detail_inference, cycles_detail_inference)]
        
        eng_dnmc_total_inference = sum(eng_dnmc)
        
        eng_stat_total_inference = pow_stat_total_inference * sum(num_cycles)

        return x, zeros_list, sizes_list, eng_dnmc_total_inference, eng_stat_total_inference, eng_dnmc_detail_inference, eng_stat_detail_inference, sum(num_cycles)

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
    
    total_cycles = None
    eng_dynamic = None
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
            
    eng_dynamic = eng_dnmc_multipliers + eng_dnmc_adders + eng_dnmc_relu + eng_dnmc_encoder + eng_dnmc_dispatcher + eng_dnmc_nbin + eng_dnmc_offset + eng_dnmc_SB + eng_dnmc_nbout + eng_dnmc_nm
    eng_dynamic_breakdown = [eng_dnmc_multipliers, eng_dnmc_adders, eng_dnmc_relu, eng_dnmc_encoder, eng_dnmc_dispatcher, eng_dnmc_nbin, eng_dnmc_offset, eng_dnmc_SB, eng_dnmc_nbout, eng_dnmc_nm]
    
    total_cycles = cycles_multipliers + cycles_adders + cycles_relu + cycles_encoder + cycles_dispatcher + cycles_nbin + cycles_offset + cycles_SB + cycles_nbout + cycles_nm
    # ORIGINAL ORDER: cycles_multipliers, cycles_adders, cycles_relu, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_offset, cycles_SB, cycles_nbout, cycles_nm
    # For the num_cycles that overlap with other component's cycles, we reflect them out as below to correctly compute the static energy of each component using eng_static_breakdown
    total_cycles_breakdown = [cycles_multipliers, cycles_multipliers, cycles_multipliers, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_nbin, cycles_nbin, cycles_encoder, cycles_nm]
    
    return eng_dynamic, total_cycles, eng_dynamic_breakdown, total_cycles_breakdown

def fc_power (input, weight, architecture):

    eng_dynamic = None
    total_cycles = None
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
            
    eng_dynamic = eng_dnmc_multipliers + eng_dnmc_adders + eng_dnmc_relu + eng_dnmc_encoder + eng_dnmc_dispatcher + eng_dnmc_nbin + eng_dnmc_offset + eng_dnmc_SB + eng_dnmc_nbout + eng_dnmc_nm
    eng_dynamic_breakdown = [eng_dnmc_multipliers, eng_dnmc_adders, eng_dnmc_relu, eng_dnmc_encoder, eng_dnmc_dispatcher, eng_dnmc_nbin, eng_dnmc_offset, eng_dnmc_SB, eng_dnmc_nbout, eng_dnmc_nm]
    
    total_cycles = cycles_multipliers + cycles_adders + cycles_relu + cycles_encoder + cycles_dispatcher + cycles_nbin + cycles_offset + cycles_SB + cycles_nbout + cycles_nm
    # ORIGINAL ORDER: cycles_multipliers, cycles_adders, cycles_relu, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_offset, cycles_SB, cycles_nbout, cycles_nm
    # For the num_cycles that overlap with other component's cycles, we reflect them out as below to correctly compute the static energy of each component using eng_static_breakdown
    total_cycles_breakdown = [cycles_multipliers, cycles_multipliers, cycles_multipliers, cycles_encoder, cycles_dispatcher, cycles_nbin, cycles_nbin, cycles_nbin, cycles_encoder, cycles_nm]
    
    return eng_dynamic, total_cycles, eng_dynamic_breakdown, total_cycles_breakdown

if __name__ == '__main__':
    
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="Customized LeNet5 Network with Fashion-MNIST Dataset")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weights', default="../2_copy_weight/lenet5_fmnist_fc_one_out.pkl", help="The path to the pre-trained weights")
    parser.add_argument('--dataset', default="../fmnist_dataset", help="The path to the Fashion-MNIST dataset")
    parser.add_argument('--power', action='store_true', help="To generate inference power statistics")
    parser.add_argument('--arch', default='cnvlutin', help="To specify the underlying architecture: cnvlutin or dadiannao")
    parser.add_argument('--adversarial', action='store_true', help="To test the adversarial dataset rather than original one")
    parser.add_argument('--im_index_first', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=10000, type=int, help="The last index of the dataset")
    parser.add_argument('--adv_images', default=None, help="The path to the adversarial images generated by energy attack algorithm")
    args = parser.parse_args()
    print(f"\n{args}\n")
    
    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{device} assigned for processing!\n")
    
    # Fashion-MNIST dataset and dataloader declaration
    normalization = transforms.Normalize(mean=0.3814, std=0.3994)
    TRANSFORM = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), normalization])
    
    if args.adversarial:
        # To load the GAN/EA dataset from the offline file generated by GAN/EA algorithm
        test_dataset = torch.load(args.adv_images, map_location=torch.device('cpu'))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        test_dataset = datasets.FashionMNIST(root=args.dataset, train=False, download=True, transform=TRANSFORM)
        test_dataset_sub = torch.utils.data.Subset(test_dataset, list(range(args.im_index_first, args.im_index_last)))
        test_loader = DataLoader(test_dataset_sub, batch_size=args.batch_size, shuffle=False)
        
    # Network Initialization
    model = LeNet5(args=args).to(device)
    
    # To load the pretrained model
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    else:
        print('No weights are provided.')
        sys.exit()

    corrects = 0
    num_processed_images = 0
    num_layers = 4
    
    total_inference_cycles = 0
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
      
        if args.adversarial:
            image = image.squeeze(0).to(device)
            label = label.squeeze(0).to(device)
        else:
            image = image.to(device)
            label = label.to(device)

        output = model(image, True, args.power)
                   
        # Sparsity calculations
        for i in range(num_layers): zeros_all_layers[i] += output[1][i]
        for i in range(num_layers): sizes_all_layers[i] += output[2][i]
        net_zeros += sum(output[1])
        net_sizes += sum(output[2])
            
        # Energy consumption profiling
        total_dynamic_energy_dataset += output[3]
        total_static_energy_dataset += output[4]
        eng_dnmc_detail_dataset = [a + b for a, b in zip(eng_dnmc_detail_dataset, output[5])]
        eng_stat_detail_dataset = [a + b for a, b in zip(eng_stat_detail_dataset, output[6])]
        total_inference_cycles += output[7]

        # Prediction and Accuracy Measurement
        _, preds = torch.max(output[0].data, 1)
        corrects += (preds == label).sum().item()
        num_processed_images += label.size(0)
        
    # To print accuracy statistics        
    accuracy = corrects / num_processed_images
    print(f"\nAccuracy of Testing is (percent): {accuracy*100:.2f}\n")
        
    # To print sparsity rate statistics
    SR = []
    for i in range(num_layers):
        SR.append((zeros_all_layers[i] / sizes_all_layers[i]) if sizes_all_layers[i] != 0 else 0)

    SR_Net = (net_zeros/net_sizes) if net_sizes != 0 else 0

    for i in range(num_layers):
        print(f"Sparsity rate of L{i+1}: {SR[i]:.5f}")
    print(f"Sparsity rate of Network: {SR_Net:.5f}")
        
    # To print energy consumption and latency statistics
    avg_inference_cycles = total_inference_cycles / num_processed_images
    avg_inference_latency = critical_path_delay * avg_inference_cycles
    avg_dynamic_energy_dataset = total_dynamic_energy_dataset / num_processed_images
    avg_static_energy_dataset = (total_static_energy_dataset / num_processed_images) * critical_path_delay
    total_energy = avg_dynamic_energy_dataset + avg_static_energy_dataset
    eng_stat_detail_dataset = [x * critical_path_delay for x in eng_stat_detail_dataset]
    eng_total_detail_dataset = [a+b for a, b in zip(eng_dnmc_detail_dataset, eng_stat_detail_dataset)]

    if args.power:
        print('\n########## Latency Statistics ##########\n')
        print('Average Inference Latency: %1.9f (Sec)' % (avg_inference_latency))
        print('Average Inference Cycles : %1.2f' % (avg_inference_cycles))
        print('\n########## Energy Statistics ##########\n')
        print('Average Dynamic Energy of Dataset: %1.9f (J)' % (avg_dynamic_energy_dataset))
        print('Average Static Energy of Dataset: %1.9f (J)' % (avg_static_energy_dataset))
        print('Total Energy of Dataset: %1.9f (J)' % (total_energy),'\n')
        print('########## Dynamic Energy Breakdown ##########\n')
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
        print('Total      (J): %1.20f' % (total_eng_dnmc),'\n')
    
        print('########## Static Energy Breakdown ###########\n')
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
        print('Total      (J): %1.20f' % (total_eng_stat),'\n')
    
        print('########## Total Energy Breakdown ############\n')
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
        print('Total      (J): %1.20f' % (total_eng),'\n')

    sys.exit(0)

# For testing original dataset
# python3 fmnist_power_delay.py --power --arch cnvlutin --batch_size 10 --dataset ../fmnist_dataset --weights ../2_copy_weight/lenet5_fmnist_fc_one_out.pkl

# For testing adversarial dataset
# python3 fmnist_power_delay.py --power --arch cnvlutin --batch_size 1 --adversarial --adv_images ../5_detection/adversarial_data/adversarial_dataset.pt --weights ../2_copy_weight/lenet5_fmnist_fc_one_out.pkl
