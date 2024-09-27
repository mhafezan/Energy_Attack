import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        
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
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10))
        
        self.my_fc1 = nn.Linear(3*256*256, 1, bias=False)
        self.my_fc2 = nn.Linear(64*31*31, 1, bias=False)
        self.my_fc3 = nn.Linear(192*15*15, 1, bias=False)
        self.my_fc4 = nn.Linear(384*15*15, 1, bias=False)
        self.my_fc5 = nn.Linear(256*15*15, 1, bias=False)

        self.my_fc6 = nn.Linear(9216, 1, bias=False)
        self.my_fc7 = nn.Linear(4096, 1, bias=False)

        self.my_fc8 = nn.Linear(7, 1, bias=False)

    def forward(self, x):
        
        # To initialize the zeros and sizes with zero
        L1_zeros, L2_zeros, L3_zeros, L4_zeros, L5_zeros, L6_zeros, L7_zeros = (0,) * 7
        L1_size, L2_size, L3_size, L4_size, L5_size, L6_size, L7_size = (0,) * 7
        
        # We don't compute SR for the first convolution input, as SR is 0% for colored images
        L1_zeros, L1_size = [0] * 2
        L1_smap = sparsity_map(x)
        x = self.features[0](x)
        x = self.features[1](x)
        x = self.features[2](x)

        L2_zeros, L2_size = sparsity_rate(x)
        L2_smap = sparsity_map(x)
        x = self.features[3](x)
        x = self.features[4](x)
        x = self.features[5](x)

        L3_zeros, L3_size = sparsity_rate(x)
        L3_smap = sparsity_map(x)
        x = self.features[6](x)
        x = self.features[7](x)

        L4_zeros, L4_size = sparsity_rate(x)
        L4_smap = sparsity_map(x)
        x = self.features[8](x)
        x = self.features[9](x)

        L5_zeros, L5_size = sparsity_rate(x)
        L5_smap = sparsity_map(x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier[0](x)
        
        L6_zeros, L6_size = sparsity_rate(x)
        L6_smap = sparsity_map(x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        
        x = self.classifier[3](x)
        
        L7_zeros, L7_size = sparsity_rate(x)
        L7_smap = sparsity_map(x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)
        
        x = self.classifier[6](x)

        # Concatenate the flattened tensors to generate a unified sparsity-map
        L1_smap_flat  = L1_smap.view(-1)
        L2_smap_flat  = L2_smap.view(-1)
        L3_smap_flat  = L3_smap.view(-1)
        L4_smap_flat  = L4_smap.view(-1)
        L5_smap_flat  = L5_smap.view(-1)
        L6_smap_flat  = L6_smap.view(-1)
        L7_smap_flat  = L7_smap.view(-1)
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat, L5_smap_flat, L6_smap_flat, L7_smap_flat], dim=0)
                
        zeros_list = [L1_zeros, L2_zeros, L3_zeros, L4_zeros, L5_zeros, L6_zeros, L7_zeros]
        sizes_list = [L1_size, L2_size, L3_size, L4_size, L5_size, L6_size, L7_size]

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
    num_layers = 7
    
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
    parser = argparse.ArgumentParser(description="AlexNet Network with CIFAR10 Dataset")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism: sparsity-map or sparsity-range")
    parser.add_argument('--weights', default="../2_copy_weight_256_256/alexnet_cifar10_fc_one_out_458624.pkl", help="The path to the pre-trained weights")
    parser.add_argument('--dataset', default="../cifar10_dataset", help="The path to the train or test datasets")
    args = parser.parse_args()
    print(f"{args}\n")
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n {device} is assigned for processing! \n")

    # CIFAR10 dataset and dataloader declaration
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
        
    # Model Initialization
    model = AlexNet().to(device)

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
        torch.save(sparsity_range, "./profile_data/sparsity_ranges.pth", pickle_protocol=4)

    sys.exit(0)

# Profile: python3 cifa10_profile.py --method sparsity-map/range --dataset ../cifar10_dataset --weights ../2_copy_weight_256_256/alexnet_cifar10_fc_one_out_458624.pkl
