import torch
import argparse
import sys
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from tqdm import tqdm

# Custom Dataset to store the adversarial data
class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

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
    num_layers = 4
    layer_inclusion_threshold = 1
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    print()
    for index, (data, target) in enumerate(tqdm(test_dataset, desc='Data Progress')):

        adversarial = 1

        data, target = data.to(device), target.to(device)
        
        batch_size = data.size(0)

        for img_idx in range (batch_size):

            single_image = data[img_idx].unsqueeze(0).to(device)
            single_target = target[img_idx].unsqueeze(0).to(device)

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
    parser = argparse.ArgumentParser(description="LeNet5 Network with Fashion-MNIST Dataset")
    parser.add_argument('--weights', default='../2_copy_weight/lenet5_fmnist_fc_one_out.pkl', help="The path to the pretrained weights")
    parser.add_argument('--adv_dataset', default=None, help="The path to the adversarial dataset")
    parser.add_argument('--profile_sparsity_range', default=None, help="The path to the sparsity-range profile")
    parser.add_argument('--profile_sparsity_map', default=None, help="The path to the sparsity-map profile")
    parser.add_argument('--sim_threshold', default=0.5, type=float, help="Initial similarity threshold")
    args = parser.parse_args()
    print(f"\n{args}\n")

    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing!\n")

    # Fashion-MNIST dataset and dataloader declaration
    # To load the offline adversarial dataset generated by sparsity-attack function
    test_dataset = torch.load(args.adv_dataset, map_location=torch.device('cpu'))

    # Network Initialization
    model = LeNet5().to(device)
    
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
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

# python3 fmnist_detect.py --adv_dataset adversarial_data/adversarial_dataset.pt --profile_sparsity_range ../4_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../4_profile/profile_data/sparsity_maps.pt --sim_threshold 0.5