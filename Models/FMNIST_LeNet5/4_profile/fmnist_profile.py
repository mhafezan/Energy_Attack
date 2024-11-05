import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import os
import sys
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
        # To compute sum of ones
        self.my_fc1 = nn.Linear(1*32*32, 1, bias=False)
        self.my_fc2 = nn.Linear(6*14*14, 1, bias=False)
        self.my_fc3 = nn.Linear(16*5*5, 1, bias=False)
        self.my_fc4 = nn.Linear(120, 1, bias=False)
        self.my_fc5 = nn.Linear(4, 1, bias=False)
    
    def forward(self, x):
        
        L1_zeros, L2_zeros, L3_zeros, L4_zeros = [0] * 4
        L1_sizes, L2_sizes, L3_sizes, L4_sizes = [0] * 4
        
        L1_zeros, L1_sizes = sparsity_rate(x)
        L1_smap = sparsity_map(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        L2_zeros, L2_sizes = sparsity_rate(x)
        L2_smap = sparsity_map(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        L3_zeros, L3_sizes = sparsity_rate(x)
        L3_smap = sparsity_map(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.view(-1, 120)
        
        L4_zeros, L4_sizes = sparsity_rate(x)
        L4_smap = sparsity_map(x)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        # Concatenate the gathered flattened tensors to generate a unified sparsity-map for entire inference
        L1_smap_flat = L1_smap.view(-1)
        L2_smap_flat = L2_smap.view(-1)
        L3_smap_flat = L3_smap.view(-1)
        L4_smap_flat = L4_smap.view(-1)
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat], dim=0)
        
        zeros_list = [L1_zeros, L2_zeros, L3_zeros, L4_zeros]
        sizes_list = [L1_sizes, L2_sizes, L3_sizes, L4_sizes]

        return x, zeros_list, sizes_list, sparsity_map_total

def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

# To compute Sparsity-Map for each layer's output
def sparsity_map(input_tensor):
    sparsity_map = torch.zeros_like(input_tensor, dtype=torch.uint8)
    zero_positions = torch.eq(input_tensor, 0)
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
    num_layers = 4
    
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
                    break
    
    print(f"\nThe sparsity map converges when diff_num_ones elements are {diff_num_ones}\n")

    return updated_maps, range_for_each_class, index



if __name__ == '__main__':

    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="LeNet5 Network with FMNIST Dataset")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism: sparsity-map or sparsity-range")
    parser.add_argument('--weights', default="../2_copy_weight/lenet5_fmnist_fc_one_out.pkl", help="The path to the pre-trained weights")
    parser.add_argument('--dataset', default="../fmnist_dataset", help="The path to the train or test datasets")
    args = parser.parse_args()
    print(f"\n{args}\n")
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing!\n")

    # Fashion-MNIST dataset and dataloader declaration
    normalization = transforms.Normalize(mean=0.3814, std=0.3994)
    TRANSFORM = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), normalization])
    train_dataset = torchvision.datasets.FashionMNIST(root=args.dataset, train=True, download=True, transform=TRANSFORM)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Model Initialization
    model = LeNet5().to(device)

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    else:
        print('No weights are provided.')
        sys.exit()
        
    # Set the model in evaluation mode
    model.eval()
        
    num_classes  = len(train_dataset.classes)

    # The profile function returns an array in which each element includes a Sparsity-Map Tensor assigned to each Class
    sparsity_maps, sparsity_range, index = Profile (model, device, train_loader, num_classes, args)
        
    # Store the Sparsity_Maps in an offline file
    print(f"We should stop profiling at data-point {index} when profiling is based on {args.method}")
    print(f"{((index+1)/len(train_loader))*100:.2f} % of training-set has been used for profiling.")
    
    if not os.path.isdir('profile_data'):
        os.mkdir('profile_data')

    if args.method == 'sparsity-map':
        # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
        torch.save(sparsity_maps, "./profile_data/sparsity_maps.pt", pickle_protocol=4)
        
    if args.method == 'sparsity-range':
        torch.save(sparsity_range, "./profile_data/sparsity_ranges.pt", pickle_protocol=4)

    sys.exit(0)

# Arguments: python3 fmnist_profile.py --method sparsity-map/range --dataset ../fmnist_dataset --weights ../2_copy_weight/lenet5_fmnist_fc_one_out.pkl