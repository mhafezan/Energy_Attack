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

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.features = nn.Sequential(
            
    	    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
    	    nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(

    	    nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
    
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
    
            nn.Linear(4096, num_classes))
        
        self.my_fc1 = nn.Linear(3*224*224, 1, bias=False)
        self.my_fc2 = nn.Linear(64*224*224, 1, bias=False)
        self.my_fc3 = nn.Linear(64*112*112, 1, bias=False)
        self.my_fc4 = nn.Linear(128*112*112, 1, bias=False)
        self.my_fc5 = nn.Linear(128*56*56, 1, bias=False)
        self.my_fc6 = nn.Linear(256*56*56, 1, bias=False)
        self.my_fc7 = nn.Linear(256*56*56, 1, bias=False)
        self.my_fc8 = nn.Linear(256*28*28, 1, bias=False)
        self.my_fc9 = nn.Linear(512*28*28, 1, bias=False)
        self.my_fc10 = nn.Linear(512*28*28, 1, bias=False)
        self.my_fc11 = nn.Linear(512*14*14, 1, bias=False)
        self.my_fc12 = nn.Linear(512*14*14, 1, bias=False)
        self.my_fc13 = nn.Linear(512*14*14, 1, bias=False)
        self.my_fc14 = nn.Linear(25088, 1, bias=False)
        self.my_fc15 = nn.Linear(4096, 1, bias=False)

        self.my_fc16 = nn.Linear(15, 1, bias=False)

    def forward(self, x):
        
        L1_zeros, L2_zeros, L3_zeros, L4_zeros, L5_zeros, L6_zeros, L7_zeros, L8_zeros, L9_zeros, L10_zeros, L11_zeros, L12_zeros, L13_zeros, L14_zeros, L15_zeros = [0] * 15
        L1_sizes, L2_sizes, L3_sizes, L4_sizes, L5_sizes, L6_sizes, L7_sizes, L8_sizes, L9_sizes, L10_sizes, L11_sizes, L12_sizes, L13_sizes, L14_sizes, L15_sizes = [0] * 15
        
        L1_zeros, L1_sizes = sparsity_rate(x)
        L1_smap = sparsity_map(x)
        x = self.features[0](x)
        x = self.features[1](x)

        L2_zeros, L2_sizes = sparsity_rate(x)
        L2_smap = sparsity_map(x)
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)

        L3_zeros, L3_sizes = sparsity_rate(x)
        L3_smap = sparsity_map(x)
        x = self.features[5](x)
        x = self.features[6](x)

        L4_zeros, L4_sizes = sparsity_rate(x)
        L4_smap = sparsity_map(x)
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)

        L5_zeros, L5_sizes = sparsity_rate(x)
        L5_smap = sparsity_map(x)
        x = self.features[10](x)
        x = self.features[11](x)

        L6_zeros, L6_sizes = sparsity_rate(x)
        L6_smap = sparsity_map(x)
        x = self.features[12](x)
        x = self.features[13](x)

        L7_zeros, L7_sizes = sparsity_rate(x)
        L7_smap = sparsity_map(x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)

        L8_zeros, L8_sizes = sparsity_rate(x)
        L8_smap = sparsity_map(x)
        x = self.features[17](x)
        x = self.features[18](x)

        L9_zeros, L9_sizes = sparsity_rate(x)
        L9_smap = sparsity_map(x)
        x = self.features[19](x)
        x = self.features[20](x)

        L10_zeros, L10_sizes = sparsity_rate(x)
        L10_smap = sparsity_map(x)
        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)

        L11_zeros, L11_sizes = sparsity_rate(x)
        L11_smap = sparsity_map(x)
        x = self.features[24](x)
        x = self.features[25](x)

        L12_zeros, L12_sizes = sparsity_rate(x)
        L12_smap = sparsity_map(x)
        x = self.features[26](x)
        x = self.features[27](x)

        L13_zeros, L13_sizes = sparsity_rate(x)
        L13_smap = sparsity_map(x)
        x = self.features[28](x)
        x = self.features[29](x)
        x = self.features[30](x)

        x = x.view(x.size(0), -1)

        L14_zeros, L14_sizes = sparsity_rate(x)
        L14_smap = sparsity_map(x)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)

        L15_zeros, L15_sizes = sparsity_rate(x)
        L15_smap = sparsity_map(x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)

        x = self.classifier[6](x)
        
        # Concatenate the gathered flattened tensors to generate a unified sparsity-map for entire inference
        L1_smap_flat = L1_smap.view(-1)
        L2_smap_flat = L2_smap.view(-1)
        L3_smap_flat = L3_smap.view(-1)
        L4_smap_flat = L4_smap.view(-1)
        L5_smap_flat = L5_smap.view(-1)
        L6_smap_flat = L6_smap.view(-1)
        L7_smap_flat = L7_smap.view(-1)
        L8_smap_flat = L8_smap.view(-1)
        L9_smap_flat = L9_smap.view(-1)
        L10_smap_flat = L10_smap.view(-1)
        L11_smap_flat = L11_smap.view(-1)
        L12_smap_flat = L12_smap.view(-1)
        L13_smap_flat = L13_smap.view(-1)
        L14_smap_flat = L14_smap.view(-1)
        L15_smap_flat = L15_smap.view(-1)
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat, L5_smap_flat, L6_smap_flat, L7_smap_flat, L8_smap_flat, L9_smap_flat, L10_smap_flat, L11_smap_flat, L12_smap_flat, L13_smap_flat, L14_smap_flat, L15_smap_flat], dim=0)
        
        zeros_list = [L1_zeros, L2_zeros, L3_zeros, L4_zeros, L5_zeros, L6_zeros, L7_zeros, L8_zeros, L9_zeros, L10_zeros, L11_zeros, L12_zeros, L13_zeros, L14_zeros, L15_zeros]
        sizes_list = [L1_sizes, L2_sizes, L3_sizes, L4_sizes, L5_sizes, L6_sizes, L7_sizes, L8_sizes, L9_sizes, L10_sizes, L11_sizes, L12_sizes, L13_sizes, L14_sizes, L15_sizes]

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
    num_layers = 15
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
    parser = argparse.ArgumentParser(description="VGG16 Network with ImageNet Dataset")
    parser.add_argument('--weights', default='../2_copy_weight/vgg16_imagenet_fc_one_out.pkl', help="The path to the pretrained weights")
    parser.add_argument('--adv_dataset', default=None, help="The path to the adversarial dataset")
    parser.add_argument('--profile_sparsity_range', default=None, help="The path to the sparsity-range dataset")
    parser.add_argument('--profile_sparsity_map', default=None, help="The path to the sparsity-map dataset")
    parser.add_argument('--sim_threshold', default=0.5, type=float, help="similarity threshold")
    args = parser.parse_args()
    print(f"\n{args}\n")

    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{device} assigned for processing!\n")

    # ImageNet dataset and dataloader declaration
    # To load the offline adversarial dataset generated by sparsity-attack function
    test_dataset = torch.load(args.adv_dataset, map_location=torch.device('cpu'))

    # Network Initialization
    model = VGG16().to(device)
    
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
# python3 vgg16_detect.py --adv_dataset adversarial_data/adv_constrained.pt --profile_sparsity_range ../4_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../4_profile/profile_data/sparsity_maps.pt --sim_threshold 0.5

# For unconstrained:
# python3 vgg16_detect.py --adv_dataset adversarial_data/adv_unconstrained.pt --profile_sparsity_range ../4_profile/profile_data/sparsity_ranges.pt --profile_sparsity_map ../4_profile/profile_data/sparsity_maps.pt --sim_threshold 0.3