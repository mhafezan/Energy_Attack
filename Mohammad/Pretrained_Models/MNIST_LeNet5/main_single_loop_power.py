import math
import torch
import argparse
import sys
import numpy as np
from torch import nn
from torch.nn import Module
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

# LeNet5 Model definition
class LeNet5(Module):
    def __init__(self, args):
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

    def forward(self, x, SR=True, PC=False):
        
        L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size = (0,) * 10
        
        pow_dnmc = []
        pow_stat = []
        num_cycles = []
        
        if PC:
            dnmc, stat, cycles = conv_power(x, self.conv1.weight, self.conv1.stride, self.conv1.padding, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L1_zeros, L1_size = sparsity_rate(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        if PC:
            dnmc, stat, cycles = conv_power(x, self.conv2.weight, self.conv2.stride, self.conv2.padding, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L2_zeros, L2_size = sparsity_rate(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.shape[0], -1)

        if PC:
            dnmc, stat, cycles = fc_power(x, self.fc1.weight, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L3_zeros, L3_size = sparsity_rate(x)
        x = self.fc1(x)
        x = self.relu3(x)

        if PC:
            dnmc, stat, cycles = fc_power(x, self.fc2.weight, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L4_zeros, L4_size = sparsity_rate(x)
        x = self.fc2(x)
        x = self.relu4(x)

        if PC:
            dnmc, stat, cycles = fc_power(x, self.fc3.weight, args.arch)
            pow_dnmc.append(dnmc)
            pow_stat.append(stat)
            num_cycles.append(cycles)
        if SR:
            L5_zeros, L5_size = sparsity_rate(x)
        x = self.fc3(x)
        x = self.relu5(x)
        
        zeros_list = [L1_zeros] + [L2_zeros] + [L3_zeros] + [L4_zeros] + [L5_zeros]
        sizes_list = [L1_size]  + [L2_size]  + [L3_size]  + [L4_size]  + [L5_size]

        return x, zeros_list, sizes_list, sum(pow_dnmc), sum(pow_stat), sum(num_cycles)

def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

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

if __name__ == '__main__':
    
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="LenNet5 Network with MNIST Dataset")
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--weights', default=None, help="The path to the saved weights. Should be specified when testing")
    parser.add_argument('--dataset', default=None, help="The path to the train and test datasets")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN architecture")
    parser.add_argument('--arch', default='cnvlutin', help="To specify the architecture running the clean/adversarial image: cnvlutin or dadiannao")
    
    args = parser.parse_args()
    print(f"{args}\n")

    # MNIST dataset and dataloader declaration
    test_dataset  = mnist.MNIST(root=args.dataset, train=False, transform=transforms.ToTensor()) # Original Testset
    test_loader   = DataLoader(test_dataset, batch_size=args.batch_size)
    train_dataset = mnist.MNIST(root=args.dataset, train=True, download=True, transform=transforms.ToTensor())
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size)

    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n A {device} is assigned for processing! \n")

    # Network Initialization
    model = LeNet5(args).to(device)
    print(f"\n{model}\n")
    
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
    critical_path_delay = 20035 * 1e-12 # Critical path delay (in picosecond) of CNVLUTIN = Multiplier(3732) + AdderTree(16179) + Relu(124)
        
    zeros_all_layers = [0] * 5
    sizes_all_layers = [0] * 5
    net_zeros  =  0
    net_sizes  =  0

    # Set the model in evaluation mode
    model.eval()
    
    for index, (image, label) in enumerate(tqdm(test_loader, desc='Data Progress')):
        image = image.to(device)
        label = label.to(device)

        output = model(image, True, args.power) # Arguments: input image, Sparisty rate activation, Power activation
                   
        # Sparsity calculations
        for i in range(5): zeros_all_layers[i] += output[1][i]
        for i in range(5): sizes_all_layers[i] += output[2][i]
        net_zeros += sum(output[1])
        net_sizes += sum(output[2])
            
        # Power consumption profiling
        total_dynamic_power_of_dataset += output[3]
        total_static_power_of_dataset += output[4]
        total_num_cycles_of_dataset += output[5]
            
        preds = torch.argmax(output[0], dim=-1)
        current_correct_num = preds == label
        correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        sample_num += current_correct_num.shape[0]
        
        if sample_num >= 100:
            break

    # To print accuracy statistics        
    acc = correct_num / sample_num
    print()
    print('Accuracy of Testing is (percent): %1.2f' % (acc*100), flush=True)
        
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
        
    # To print power consumption statistics
    avg_dynamic_power_of_dataset = total_dynamic_power_of_dataset / sample_num
    avg_static_power_of_dataset = total_static_power_of_dataset / sample_num
    avg_dynamic_energy_of_dataset = avg_dynamic_power_of_dataset * (critical_path_delay) * total_num_cycles_of_dataset
    avg_static_energy_of_dataset = avg_static_power_of_dataset * (critical_path_delay) * total_num_cycles_of_dataset
    total_energy = avg_dynamic_energy_of_dataset + avg_static_energy_of_dataset
    print()
    print('Average Dynamic Power: %1.9f (W)' % (avg_dynamic_power_of_dataset))
    print('Average Static Power: %1.9f (W)' % (avg_static_power_of_dataset))
    print('Average Dynamic Energy: %1.9f (J)' % (avg_dynamic_energy_of_dataset))
    print('Average Static Energy: %1.9f (J)' % (avg_static_energy_of_dataset))
    print('Total Energy: %1.9f (J)' % (total_energy))

# Arguments: python3 main_single_loop_power.py --power --arch cnvlutin --batch_size 10 --dataset ./test --weights models/mnist_0.9876.pkl