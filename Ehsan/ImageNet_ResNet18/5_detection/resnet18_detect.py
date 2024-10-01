import torch
import argparse
import sys
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

# Custom Dataset to store the adversarial data
class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
        L1_sizes = 0
        L2_L3_zeros, L4_L5_zeros, L6_L8_zeros, L9_L10_zeros, L11_L13_zeros, L14_L15_zeros, L16_L18_zeros, L19_L20_zeros = ([] for _ in range(8))
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
    maps = torch.zeros_like(input_tensor, dtype=torch.uint8)
    zero_positions = torch.eq(input_tensor, 0)
    maps = maps.masked_fill(zero_positions, 1)
    return maps

# The output ranges from 0 to 1, where 0 means no similarity and 1 means identical tensors (all non-zero elements are the same)
def jaccard_similarity(tensor1, tensor2):
    intersection = torch.logical_and(tensor1, tensor2).sum().item()
    union = torch.logical_or(tensor1, tensor2).sum().item()
    return intersection / union

def Sparsity_Attack_Detection (model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes, args):
    
    generated_adversarial_for_class = [0] * num_classes 
    num_of_items_in_class = [0] * num_classes

    # Initialization for sparsity-maps
    correctly_predicted_adversarial_for_class_map = [0] * num_classes
    correctly_predicted_adversarial_ratio_map = []

    # Initialization for sparsity-ranges
    num_layers = 20
    layer_inclusion_threshold = 1
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    for index, (data, target) in enumerate(tqdm(test_dataset, desc='Data Progress')):

        adversarial = 1

        data = data.to(device)
        target = target.to(device)           
        
        batch_size = data.size(0)

        for i in range (batch_size):

            single_image = data[i].unsqueeze(0).to(device)
            single_target = target[i].unsqueeze(0).to(device)

            output = model(single_image)
            
            pred = output[0].max(1, keepdim=False)[1]

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
            
            sim_rate = jaccard_similarity(offline_sparsity_maps[pred.item()], output[3])

            # if the following condition is True, we predict that the input is adversarial
            if sim_rate <= args.sim_threshold: 
                if adversarial == 1:
                    correctly_predicted_adversarial_for_class_map[single_target.item()] += 1

            ###################################################################################################################################

            if adversarial == 1:
                generated_adversarial_for_class[single_target.item()] += 1 

            num_of_items_in_class[single_target.item()] += 1
        
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
    parser = argparse.ArgumentParser(description="ResNet18 Network with ImageNet Dataset")
    parser.add_argument('--weights', default='../2_copy_weight/resnet18_imagenet_fc_one_out.pkl', help="The path to the copied weights")
    parser.add_argument('--adv_dataset', default=None, help="The path to the adversarial dataset")
    parser.add_argument('--profile_sparsity_range', default=None, help="The path to the sparsity-range offline profile")
    parser.add_argument('--profile_sparsity_map', default=None, help="The path to the sparsity-map offline profile")
    parser.add_argument('--sim_threshold', default=0.5, type=float, help="similarity threshold")
    args = parser.parse_args()
    print(f"\n{args}\n")

    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{device} assigned for processing!\n")

    # To load the offline adversarial dataset generated by sparsity-attack function
    test_dataset = torch.load(args.adv_dataset, map_location=torch.device('cpu'))

    # Model Initialization
    model = ResNet18(BasicBlock, [2, 2, 2, 2]).to(device)
    
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()
        
    # Set the model in evaluation mode
    model.eval()

    # Load the sparsity-maps and sparsity-ranges from offline file (generated by profiling phase)
    offline_sparsity_ranges = torch.load(args.profile_sparsity_range, map_location=device)
    offline_sparsity_maps = torch.load(args.profile_sparsity_map, map_location=device)
        
    num_classes  = len(offline_sparsity_maps)

    # Prints the number of detected adversarials in each class
    Sparsity_Attack_Detection (model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes, args)
    
    sys.exit(0)

# For constrained:
# python3 resnet18_detect.py --adv_dataset adversarial_data/adv_constrained.pt --profile_sparsity_range ../4_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../4_profile/profile_data/sparsity_maps.pt --sim_threshold 0.5

# For unconstrained:
# python3 resnet18_detect.py --adv_dataset adversarial_data/adv_unconstrained.pt --profile_sparsity_range ../4_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../4_profile/profile_data/sparsity_maps.pt --sim_threshold 0.3
