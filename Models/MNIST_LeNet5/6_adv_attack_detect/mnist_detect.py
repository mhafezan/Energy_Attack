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

# LeNet5 Model definition
class LeNet5 (nn.Module):
    def __init__(self):
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
       
        self.my_fc1 = nn.Linear(1*28*28, 1, bias=False)
        self.my_fc2 = nn.Linear(6*12*12, 1, bias=False)
        self.my_fc3 = nn.Linear(256, 1, bias=False)
        self.my_fc4 = nn.Linear(120, 1, bias=False)
        self.my_fc5 = nn.Linear(84, 1, bias=False)

        self.my_fc6 = nn.Linear(5, 1, bias=False)
        
    def forward (self, x):
        
        L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size = (0,) * 10 
       
        L1_zeros, L1_size = sparsity_rate(x)
        L1_smap = sparsity_map(x)
        x = self.conv1(x)
        x = self.relu1(x)    
        x = self.pool1(x)
        
        L2_zeros, L2_size = sparsity_rate(x)
        L2_smap = sparsity_map(x)
        x = self.conv2(x)
        x = self.relu2(x)      
        x = self.pool2(x)
        
        x = x.view(x.shape[0], -1)
      
        L3_zeros, L3_size = sparsity_rate(x)
        L3_smap = sparsity_map(x)
        x = self.fc1(x)
        x = self.relu3(x)

        L4_zeros, L4_size = sparsity_rate(x)
        L4_smap = sparsity_map(x)
        x = self.fc2(x)
        x = self.relu4(x)
       
        L5_zeros, L5_size = sparsity_rate(x)
        L5_smap = sparsity_map(x)
        x = self.fc3(x)
        x = self.relu5(x)
        
        zeros_list = [L1_zeros] + [L2_zeros] + [L3_zeros] + [L4_zeros] + [L5_zeros]
        sizes_list = [L1_size]  + [L2_size]  + [L3_size]  + [L4_size]  + [L5_size]
        
        # Concatenate the flattened tensors to generate a Unified Sparsity-Map
        L1_smap_flat = L1_smap.view(-1)
        L2_smap_flat = L2_smap.view(-1)
        L3_smap_flat = L3_smap.view(-1)
        L4_smap_flat = L4_smap.view(-1)
        L5_smap_flat = L5_smap.view(-1)
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat, L5_smap_flat], dim=0)
        
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
    num_layers = 5
    layer_inclusion_threshold = num_layers - 4
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    for index, (data, target, adversarial) in enumerate(tqdm(test_dataset, desc='Data Progress')): 
        
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
    parser = argparse.ArgumentParser(description="LenNet5 Network with MNIST Dataset")
    parser.add_argument('--weights', default='../2_copy_weight/lenet5_mnist_fc_one_out.pkl', help="The path to the pretrained weights")
    parser.add_argument('--adv_dataset', default=None, help="The path to the adversarial dataset")
    parser.add_argument('--profile_sparsity_range', default=None, help="The path to the sparsity-range dataset")
    parser.add_argument('--profile_sparsity_map', default=None, help="The path to the sparsity-map dataset")
    parser.add_argument('--sim_threshold', default=0.5, type=float, help="sim_threshold, default mnist_unconstrained=0.3, default mnist_unconstrained=0.5, clean=0.5")
    args = parser.parse_args()
    print(f"{args}\n")

    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n A {device} is assigned for processing! \n")

    # MNIST dataset and dataloader declaration
    # To load the adversarial_dataset from the offline file generated by sparsity-attack function (50% of 10K testset are polluted)
    test_dataset = torch.load(args.adv_dataset, map_location=device)

    # Network Initialization
    model = LeNet5().to(device)
    print(f"\n{model}\n")
    
    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()
        
    # Set the model in evaluation mode
    model.eval()

    # Load the sparsity-maps and sparsity-ranges from offline file (generated by profiling phase)
    offline_sparsity_ranges = torch.load(args.profile_sparsity_range)
    offline_sparsity_maps = torch.load(args.profile_sparsity_map)
        
    num_classes  = len(offline_sparsity_maps) # or length of (offline_sparsity_ranges)

    # Prints the number of detected adversarials in each class
    Sparsity_Attack_Detection (model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes, args)



# for constrained:
# python3 mnist_detect.py --adv_dataset adversarial_data/adv_constrained.pt --profile_sparsity_range ../5_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../5_profile/profile_data/sparsity_maps.pt --sim_threshold 0.5

# for unconstrained:
# python3 mnist_detect.py --adv_dataset adversarial_data/adv_unconstrained.pt --profile_sparsity_range ../5_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../5_profile/profile_data/sparsity_maps.pt --sim_threshold 0.3