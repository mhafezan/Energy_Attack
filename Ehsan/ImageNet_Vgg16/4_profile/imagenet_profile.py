import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    num_layers = 15
    
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
    parser = argparse.ArgumentParser(description="VGG16 Network with ImageNet Dataset")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism: sparsity-map or sparsity-range")
    parser.add_argument('--weights', default="../2_copy_weight/vgg16_imagenet_fc_one_out.pkl", help="The path to the model pre-trained weights")
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
    model = VGG16().to(device)

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

# Profile: python3 imagenet_profile.py --method sparsity-map/range --dataset ../../Imagenet_dataset --weights ../2_copy_weight/vgg16_imagenet_fc_one_out.pkl