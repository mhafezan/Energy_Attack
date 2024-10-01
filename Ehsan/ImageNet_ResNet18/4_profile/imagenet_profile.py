import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, inplanes, planes, fc_in, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    	# To compute sum of ones
        if self.downsample is not None:
            self.my_fc = nn.ModuleList([
                nn.Linear((planes//stride)*fc_in*fc_in, 1, bias=False),
                nn.Linear(planes*(fc_in//stride)*(fc_in//stride), 1, bias=False),
                nn.Linear((planes//stride)*fc_in*fc_in, 1, bias=False),
                nn.Linear(3, 1, bias=False)])
        else:
            self.my_fc = nn.ModuleList([
                nn.Linear(planes*(fc_in//stride)*(fc_in//stride), 1, bias=False),
                nn.Linear(planes*(fc_in//stride)*(fc_in//stride), 1, bias=False),
                nn.Linear(2, 1, bias=False)])

    def forward(self, x):
        
        zeros = []
        sizes = []
        smaps = []
        
        residual = x

        b1_zeros, b1_sizes = sparsity_rate(x)
        zeros.append(b1_zeros)
        sizes.append(b1_sizes)
        smaps.append(sparsity_map(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        b2_zeros, b2_sizes = sparsity_rate(out)
        zeros.append(b2_zeros)
        sizes.append(b2_sizes)
        smaps.append(sparsity_map(out))
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            b3_zeros, b3_sizes = sparsity_rate(x)
            zeros.append(b3_zeros)
            sizes.append(b3_sizes)
            smaps.append(sparsity_map(x))
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, zeros, sizes, smaps

class ResNet18(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block,  64, layers[0], 56, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], 56, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], 28, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], 14, stride=2)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    	# To compute number of ones
        self.my_fc_0  = nn.Linear(3*224*224, 1, bias=False)
        self.my_fc_9 = nn.Linear(9, 1, bias=False)

    def _make_layer(self, block, planes, blocks, fc_in, stride=1):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                         nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, fc_in, stride, downsample))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, fc_in=fc_in//stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        # To initialize the zeros and sizes with zero value
        L1_zeros = 0
        L2_L3_zeros, L4_L5_zeros, L6_L8_zeros, L9_L10_zeros, L11_L13_zeros, L14_L15_zeros, L16_L18_zeros, L19_L20_zeros = ([] for _ in range(8))
        L1_sizes = 0
        L2_L3_sizes, L4_L5_sizes, L6_L8_sizes, L9_L10_sizes, L11_L13_sizes, L14_L15_sizes, L16_L18_sizes, L19_L20_sizes = ([] for _ in range(8))
        
        L1_zeros, L1_sizes = sparsity_rate(x)
        L1_smap = sparsity_map(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x, L2_L3_zeros, L2_L3_sizes, L2_L3_smap = self.layer1[0](x)
        x, L4_L5_zeros, L4_L5_sizes, L4_L5_smap = self.layer1[1](x)
        x, L6_L8_zeros, L6_L8_sizes, L6_L8_smap = self.layer2[0](x)
        x, L9_L10_zeros, L9_L10_sizes, L9_L10_smap = self.layer2[1](x)
        x, L11_L13_zeros, L11_L13_sizes, L11_L13_smap = self.layer3[0](x)
        x, L14_L15_zeros, L14_L15_sizes, L14_L15_smap = self.layer3[1](x)
        x, L16_L18_zeros, L16_L18_sizes, L16_L18_smap = self.layer4[0](x)
        x, L19_L20_zeros, L19_L20_sizes, L19_L20_smap = self.layer4[1](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Concatenate the gathered flattened tensors to generate a unified sparsity-map for entire inference
        L1_smap_flat = L1_smap.view(-1)
        L2_L3_smap_flat   = [tensor.view(-1) for tensor in L2_L3_smap]
        L4_L5_smap_flat   = [tensor.view(-1) for tensor in L4_L5_smap]
        L6_L8_smap_flat   = [tensor.view(-1) for tensor in L6_L8_smap]
        L9_L10_smap_flat  = [tensor.view(-1) for tensor in L9_L10_smap]
        L11_L13_smap_flat = [tensor.view(-1) for tensor in L11_L13_smap]
        L14_L15_smap_flat = [tensor.view(-1) for tensor in L14_L15_smap]
        L16_L18_smap_flat = [tensor.view(-1) for tensor in L16_L18_smap]
        L19_L20_smap_flat = [tensor.view(-1) for tensor in L19_L20_smap]
        smap_total = torch.cat([L1_smap_flat] + L2_L3_smap_flat + L4_L5_smap_flat + L6_L8_smap_flat + L9_L10_smap_flat + L11_L13_smap_flat + L14_L15_smap_flat + L16_L18_smap_flat + L16_L18_smap_flat + L19_L20_smap_flat, dim=0)
        
        zeros_list = [L1_zeros] + L2_L3_zeros + L4_L5_zeros + L6_L8_zeros + L9_L10_zeros + L11_L13_zeros + L14_L15_zeros + L16_L18_zeros + L19_L20_zeros
        sizes_list = [L1_sizes] + L2_L3_sizes + L4_L5_sizes + L6_L8_sizes + L9_L10_sizes + L11_L13_sizes + L14_L15_sizes + L16_L18_sizes + L19_L20_sizes

        return x, zeros_list, sizes_list, smap_total

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
    num_layers = 20
    
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
    parser = argparse.ArgumentParser(description="ResNet18 Network with ImageNet Dataset")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism: sparsity-map or sparsity-range")
    parser.add_argument('--weights', default="../2_copy_weight/resnet18_imagenet_fc_one_out.pkl", help="The path to the model pre-trained weights")
    parser.add_argument('--dataset', default="../../Imagenet_dataset", help="The path to the train or test datasets")
    args = parser.parse_args()
    print(f"\n{args}\n")
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing!\n")

    # ImageNet dataset and dataloader declaration
    train_dir = os.path.join(args.dataset, 'ILSVRC2012_img_train')
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalization])
    """transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization])"""
    train_dataset = datasets.ImageFolder(train_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Model Initialization
    model = ResNet18(BasicBlock, [2, 2, 2, 2]).to(device)

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()
        
    # Set the model in evaluation mode
    model.eval()
        
    num_classes  = 1000

    # The profile function returns an array in which each element comprises a Sparsity-Map Tensor assigned to each Class
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

# Profile: python3 imagenet_profile.py --method sparsity-map/range --dataset ../../Imagenet_dataset --weights ../2_copy_weight/resnet18_imagenet_fc_one_out.pkl