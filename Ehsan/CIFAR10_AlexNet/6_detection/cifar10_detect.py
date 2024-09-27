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
    maps = torch.zeros_like(input_tensor, dtype=torch.uint8)
    # To find the positions where the input tensor is zero
    zero_positions = torch.eq(input_tensor, 0)
    # To set the corresponding positions in the bit_map to 1
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
    num_layers = 7
    layer_inclusion_threshold = num_layers - 6
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    for index, (data, target) in enumerate(tqdm(test_dataset, desc='Data Progress')):

        adversarial = 1

        data = data.to(device)
        target =  target.to(device)           
        
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
    parser = argparse.ArgumentParser(description="Alexnet Network with CIFAR10 Dataset")
    parser.add_argument('--weights', default='../2_copy_weight_256_256/alexnet_cifar10_fc_one_out_458624.pkl', help="The path to the pretrained weights")
    parser.add_argument('--adv_dataset', default=None, help="The path to the adversarial dataset")
    parser.add_argument('--profile_sparsity_range', default=None, help="The path to the sparsity-range dataset")
    parser.add_argument('--profile_sparsity_map', default=None, help="The path to the sparsity-map dataset")
    parser.add_argument('--sim_threshold', default=0.5, type=float, help="similarity threshold")
    args = parser.parse_args()
    print(f"{args}\n")

    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n A {device} is assigned for processing! \n")

    # CIFAR10 dataset and dataloader declaration
    # To load the adversarial_dataset from the offline file generated by sparsity-attack function
    test_dataset = torch.load(args.adv_dataset, map_location=torch.device('cpu'))#, map_location=device)

    # Network Initialization
    model = AlexNet().to(device)
    print(f"\n{model}\n")
    
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
        
    num_classes  = len(offline_sparsity_maps) # or length of (offline_sparsity_ranges)

    # Prints the number of detected adversarials in each class
    Sparsity_Attack_Detection (model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes, args)
    
    sys.exit(0)

# for constrained:
# python3 cifar10_detect.py --adv_dataset adversarial_data/adv_constrained.pt --profile_sparsity_range ../5_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../5_profile/profile_data/sparsity_maps.pt --sim_threshold 0.5

# for unconstrained:
# python3 cifar10_detect.py --adv_dataset adversarial_data/adv_unconstrained.pt --profile_sparsity_range ../5_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../5_profile/profile_data/sparsity_maps.pt --sim_threshold 0.3