import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
import sys
import argparse
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

classes = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# Custom Dataset to store the adversarial data
class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, beta=15):
        super(AlexNet, self).__init__()
        self.beta = beta
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64), # Newly Added
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192), # Newly Added
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384), # Newly Added
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # Newly Added
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # Newly Added
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # We break down the x = self.features(x) to minor modules
        x = self.features[0](x)
        x = self.features[1](x) # Newly Added
        x = self.features[2](x)
        x = self.features[3](x)
        L1_zeros, L1_size = sparsity(x)
        L1_tanh = tanh(x, self.beta)
        L1_smap = sparsity_map(x)

        x = self.features[4](x)
        x = self.features[5](x) # Newly Added
        x = self.features[6](x)
        x = self.features[7](x)
        L2_zeros, L2_size = sparsity(x)
        L2_tanh = tanh(x, self.beta)
        L2_smap = sparsity_map(x)

        x = self.features[8](x)
        x = self.features[9](x) # Newly Added
        x = self.features[10](x)
        L3_zeros, L3_size = sparsity(x)
        L3_tanh = tanh(x, self.beta)
        L3_smap = sparsity_map(x)

        x = self.features[11](x)
        x = self.features[12](x) # Newly Added
        x = self.features[13](x)
        L4_zeros, L4_size = sparsity(x)
        L4_tanh = tanh(x, self.beta)
        L4_smap = sparsity_map(x)

        x = self.features[14](x)
        x = self.features[15](x) # Newly Added
        x = self.features[16](x)
        x = self.features[17](x)
        L5_zeros, L5_size = sparsity(x)
        L5_tanh = tanh(x, self.beta)
        L5_smap = sparsity_map(x)
        
        x = self.avgpool(x)
        L6_zeros, L6_size = sparsity(x)
        L6_tanh = tanh(x, self.beta)
        L6_smap = sparsity_map(x)

        x = x.view(x.size(0), 256 * 6 * 6)

        # We break down the x = self.classifier(x) to minor modules
        x = self.classifier[0](x)
        L7_zeros, L7_size = sparsity(x)
        L7_tanh = tanh(x, self.beta)
        L7_smap = sparsity_map(x)

        x = self.classifier[1](x)
        x = self.classifier[2](x)
        L8_zeros, L8_size = sparsity(x)
        L8_tanh = tanh(x, self.beta)
        L8_smap = sparsity_map(x)

        x = self.classifier[3](x)
        L9_zeros, L9_size = sparsity(x)
        L9_tanh = tanh(x, self.beta)
        L9_smap = sparsity_map(x)

        x = self.classifier[4](x)
        x = self.classifier[5](x)
        L10_zeros, L10_size = sparsity(x)
        L10_tanh = tanh(x, self.beta)
        L10_smap = sparsity_map(x)

        x = self.classifier[6](x)

        # Adds all the tanh outputs, negates them, and finally divides them by number_of_neurons in the network
        tanh_total = (L1_tanh + L2_tanh + L3_tanh + L4_tanh + L5_tanh + L6_tanh + L7_tanh + L8_tanh + L9_tanh + L10_tanh)
        tanh_total = (-tanh_total)/(L1_size + L2_size + L3_size + L4_size + L5_size + L6_size + L7_size + L8_size + L9_size + L10_size)

        # Concatenate the flattened tensors to generate a unified sparsity-map
        L1_smap_flat  = L1_smap.view(-1)
        L2_smap_flat  = L2_smap.view(-1)
        L3_smap_flat  = L3_smap.view(-1)
        L4_smap_flat  = L4_smap.view(-1)
        L5_smap_flat  = L5_smap.view(-1)
        L6_smap_flat  = L6_smap.view(-1)
        L7_smap_flat  = L7_smap.view(-1)
        L8_smap_flat  = L8_smap.view(-1)
        L9_smap_flat  = L9_smap.view(-1)
        L10_smap_flat = L10_smap.view(-1)
        sparsity_map_total = torch.cat([L1_smap_flat, L2_smap_flat, L3_smap_flat, L4_smap_flat, L5_smap_flat, L6_smap_flat, L7_smap_flat, L8_smap_flat, L9_smap_flat,\
                                        L10_smap_flat], dim=0)
        
        zeros_list = [L1_zeros, L2_zeros, L3_zeros, L4_zeros, L5_zeros, L6_zeros, L7_zeros, L8_zeros, L9_zeros, L10_zeros]
        sizes_list = [L1_size, L2_size, L3_size, L4_size, L5_size, L6_size, L7_size, L8_size, L9_size, L10_size]

        return x, zeros_list, sizes_list, tanh_total, sparsity_map_total

def sparsity(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

# To compute tanh with beta parameter getting input_tensor
def tanh(input_tensor, beta):
    # Define beta value
    BETA = beta
    # Scale the tensor by BETA
    scaled_tensor = BETA * input_tensor
    # Apply tanh function to the scaled tensor
    output = torch.tanh(scaled_tensor)
    # Sum the activations of the tensor
    sum_of_activations = output.sum()
    return sum_of_activations

# To compute Sparsity-Map for each layer's output
def sparsity_map(input_tensor):
    # To create a tensor with all zeros and with the same shape of input tensor
    sparsity_map = torch.zeros_like(input_tensor, dtype=torch.uint8)
    # To find the positions where the input tensor is zero
    zero_positions = torch.eq(input_tensor, 0)
    # To set the corresponding positions in the bit_map to 1
    sparsity_map = sparsity_map.masked_fill(zero_positions, 1)
    return sparsity_map

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]
def clip_tensor(input_tensor, eps):
    tensor_norm = torch.norm(input_tensor, p=2)
    max_norm = eps

    # If the L2 norm is greater than the eps, we scale down the tensor by multiplying it with a factor (max_norm / tensor_norm)
    if tensor_norm > max_norm:
        clipped_tensor = input_tensor * (max_norm / tensor_norm)
    else:
        clipped_tensor = input_tensor

    return torch.clamp(clipped_tensor, 0, 1)

def calculate_range(current_sparsity_list, sparsity_range, class_index):
    
    for layer in range (len(current_sparsity_list)):

        if current_sparsity_list[layer] < sparsity_range[class_index][layer][0]:
            sparsity_range[class_index][layer][0] = current_sparsity_list[layer]

        if current_sparsity_list[layer] > sparsity_range[class_index][layer][1]:
            sparsity_range[class_index][layer][1] = current_sparsity_list[layer]

    return sparsity_range

# The output ranges from 0 to 1, where 0 means no similarity and 1 means identical tensors (all non-zero elements are the same)
def jaccard_similarity(tensor1, tensor2):
    intersection = torch.logical_and(tensor1, tensor2).sum().item()
    union = torch.logical_or(tensor1, tensor2).sum().item()
    return intersection / union

# Generates sparsity attack for P2=50% of each class in test_dataset
def Sparsity_Attack(model, device, test_loader, num_classes, c_init, args):
    
    coeff = c_init
    correct_after = 0
    correct_before = 0
    L2_Norms = []
    net_zeros_before_total, net_sizes_before_total = [0] * 2
    net_zeros_after_total, net_sizes_after_total = [0] * 2

    # A value of 1 means the sparsity attack has polluted the last processed input data of the same class, so we need to leave the current input clean.
    # A value of 0 means the sparsity attack has left the last processed input data of the same class clean, so we need to pollute the current input.
    last_status_of_class  = [0] * num_classes
    num_of_items_in_class = [0] * num_classes
    num_of_adver_in_class = [0] * num_classes
    adversarial_data = []

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):

        data, target = data.to(device), target.to(device)
        i_max = args.imax
        c_min = 0
        c_max = 1
        eps = args.eps * 1000
        eps_iter = args.eps_iter * 1000

        output = model(data)
        Net_zeros = sum(output[1])
        Net_sizes = sum(output[2])
        net_zeros_before_total += Net_zeros
        net_sizes_before_total += Net_sizes
        init_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability for Clean Input
        if init_pred.item() == target.item():
            correct_before += 1

        x = data
        for i in range(i_max):
            # Sets requires_grad attribute of X to True to compute gradients with respect to the input data
            x = x.clone().detach().requires_grad_(True)

            output = model(x)
            l_ce = F.cross_entropy(output[0], target)
            l_sparsity = output[3] # To compute SR considering Tanh function
            l_x = l_sparsity + (coeff * l_ce)
                    
            optimizer.zero_grad()
            l_x.backward()
                
            # To compute gradient of Loss function (l_x) w.r.t input x
            # Below line is used instead of g(i) = μg(i−1) + x.grad.data, because optimizer has already been set as SGD with momentum of 0.9
            g = x.grad.data
            x_new = x - (eps_iter * (g/torch.norm(g, p=2)))
            if args.constrained:
                x_new = clip_tensor(x_new, eps)

            output = model(x_new)
            final_pred = output[0].max(1, keepdim=True)[1]
            if final_pred.item() != init_pred.item():
                coeff = (coeff + c_max)/2
                x = x_new
            else:
                coeff = (coeff + c_min)/2
                x = x_new

        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of P2=50%
        if last_status_of_class[target.item()] == 0:
            if args.store_attack:
                adversarial_data.append((x_new, target, 1))
            last_status_of_class[target.item()] = 1
            num_of_items_in_class[target.item()] = num_of_items_in_class[target.item()] + 1
            num_of_adver_in_class[target.item()] = num_of_adver_in_class[target.item()] + 1
        else:
            if args.store_attack:
                adversarial_data.append((data, target, 0)) 
            last_status_of_class[target.item()] = 0
            num_of_items_in_class[target.item()] = num_of_items_in_class[target.item()] + 1

        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-data), p=2)
        L2_Norms.append(l2norm_diff)

        # Final prediction after applying attack (The two following lines can be removed)
        output = model(x_new)
        final_pred = output[0].max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct_after += 1

        # Re-compute the sparsity rate using the perturbed input
        Net_zeros = sum(output[1])
        Net_sizes = sum(output[2])
        net_zeros_after_total += Net_zeros
        net_sizes_after_total += Net_sizes

    # To Create a new dataset using the AdversarialDataset class
    adversarial_dataset = AdversarialDataset(adversarial_data)
    print(f"Distribution of data for each class: {num_of_items_in_class}")
    print(f"Number of generated adversarials in each class: {num_of_adver_in_class}")

    # Calculate overal accuracy of all test data after sparsity attack
    first_acc = correct_before/float(len(test_loader))
    final_acc = correct_after/float(len(test_loader))

    return adversarial_dataset, first_acc, final_acc, L2_Norms, (net_zeros_before_total/net_sizes_before_total), (net_zeros_after_total/net_sizes_after_total)

def Profile (model, device, train_loader, num_classes, args):
    
    updated_maps  = [None] * num_classes
    diff_num_ones = [0] * num_classes
    prev_num_ones = [0] * num_classes
    curr_num_ones = [0] * num_classes
    num_layers = 10
    
    # To define sparsity-range for each class
    range_for_each_class = [[[float('inf'), float('-inf')] for _ in range(num_layers)] for _ in range(num_classes)]

    for index, (data, target) in enumerate(tqdm(train_loader, desc='Data Progress')):
        
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)

        pred = output[0].max(1, keepdim=True)[1]
        if pred.item() == target.item():
            if args.method == 'sparsity-range':
                current_sparsity_rate = [0.0] * num_layers  # Initialize the current_sparsity list reading each model input
                for i in range(num_layers):
                    current_sparsity_rate[i] = output[1][i] / output[2][i]
                range_for_each_class = calculate_range(current_sparsity_rate, range_for_each_class, target.item())

            elif args.method == 'sparsity-map':
                if updated_maps[target.item()] == None:
                    updated_maps[target.item()] = output[4]
                    prev_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item() # Computes the number of 1s in the updated_maps
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                else:
                    updated_maps [target.item()] = torch.bitwise_or(updated_maps[target.item()], output[4]) # output[4] is the Sparsity-Map associated with the current input
                    curr_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()]) # Computes the difference between the number of 1s in the current and previous Sparsity-Maps
                    prev_num_ones[target.item()] = curr_num_ones[target.item()]

                if all(num < 1 for num in diff_num_ones):
                    print(diff_num_ones)
                    break

    return updated_maps, range_for_each_class, index

def Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes):
    
    generated_adversarial_for_class = [0] * num_classes 
    num_of_items_in_class = [0] * num_classes

    # Initialization for sparsity-maps
    sim_threshold = 0.468
    correctly_predicted_adversarial_for_class_map = [0] * num_classes
    correctly_predicted_adversarial_ratio_map = []

    # Initialization for sparsity-ranges
    num_layers = 10 # Assuming 10 layers for AlexNet
    layer_inclusion_threshold = num_layers - 9
    correctly_predicted_adversarial_for_class_range = [0] * num_classes
    correctly_predicted_adversarial_ratio_range = []

    for index, (data, target, adversarial) in enumerate(tqdm(test_dataset, desc='Data Progress')):
        
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)
        
        pred = output[0].max(1, keepdim=True)[1]

        ############################################## sparsity-range #####################################################################

        current_sparsity_rate = [0.0] * num_layers  # To construct the current_sparsity list reading each data input
        for L in range(num_layers):
            current_sparsity_rate[L] = output[1][L] / output[2][L]
        
        in_range_status = [0] * num_layers
        for M in range(num_layers):
            if not offline_sparsity_ranges[pred.item()][M][0] <= current_sparsity_rate[M] <= offline_sparsity_ranges[pred.item()][M][1]:
                in_range_status[M] = 1
        
        if sum(in_range_status) >= layer_inclusion_threshold:
            if adversarial == 1:
                correctly_predicted_adversarial_for_class_range[target.item()] += 1
        
        ############################################## sparsity-map #######################################################################
        
        sim_rate = jaccard_similarity(offline_sparsity_maps[pred.item()], output[4])

        # If sim_rate was not more than a specific threshold (i.e., sim_threshold), we predict that the input is adversarial
        if sim_rate <= sim_threshold:
            if adversarial == 1:
                correctly_predicted_adversarial_for_class_map[target.item()] += 1

        # To check the real adversarial status for the same targeted class
        if adversarial == 1:
            generated_adversarial_for_class[target.item()] += 1 

        num_of_items_in_class[target.item()] += 1

        ###################################################################################################################################

    for predicted, generated in zip(correctly_predicted_adversarial_for_class_map, generated_adversarial_for_class):
        correctly_predicted_adversarial_ratio_map.append((predicted/generated)*100)

    for predicted, generated in zip(correctly_predicted_adversarial_for_class_range, generated_adversarial_for_class):
        correctly_predicted_adversarial_ratio_range.append((predicted/generated)*100)

    correctly_predicted_adversarial_ratio_map = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_map]

    correctly_predicted_adversarial_ratio_range = ["{:.2f}".format(ratio) for ratio in correctly_predicted_adversarial_ratio_range]

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
    parser = argparse.ArgumentParser(description="AlexNet Network with CIFAR10 Dataset")
    parser.add_argument('--epochs', default=7, type=int, help="Achieves a testing accuracy of 85.07% with only 7 epochs")
    parser.add_argument('--batch_size', default=4, type=int, help="Batch size of 4 is a good choice for Training")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--weights', default=None, help="The path to the saved weights")
    parser.add_argument('--phase', default='train', help="train, test, profile, Sparsity-Attack, Sparsity-Detect")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism based on sparsity-map or sparsity-range")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon in each inner loop iteration")
    parser.add_argument('--constrained', action='store_true', help="To enable clipping the generated purturbed data")
    parser.add_argument('--store_attack', action='store_true', help="To enable or disable algorithm to store generated adversarials in an output dataset")
    args = parser.parse_args()
    print(args)

    # CIFAR10 dataset and dataloader declaration
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.phase == 'Sparsity-Attack':
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    elif args.phase == 'profile':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
    elif args.phase == 'Sparsity-Detect':
        # To load the adversarial_dataset from the offline file generated by Sparsity-Attack function (50% are polluted)
        test_dataset = torch.load('./adversarial_data/adversarial_dataset_unconstrained.pt', map_location=torch.device('cpu'))
    else:
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # To use the genuine AlexNet with 1000 classes as the Network Model
    model = AlexNet(num_classes=1000, beta=args.beta)
    print(model)

    # To modify the AlexNet to accept only 10 classes and remove overfitting
    model.classifier[1] = nn.Linear(9216,4096)
    model.classifier[4] = nn.Linear(4096,1024)
    model.classifier[6] = nn.Linear(1024,10)
    print(model)

    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\n A {device} is assigned for processing! \n")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.phase == 'train':
        # Training
        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model(inputs)
                loss = loss_fn(output[0], labels) # mohammad: added [0] for compatibility with sparsity calculations
                loss.backward()
                optimizer.step()

                # Time
                end_time = time.time()
                time_taken = end_time - start_time

                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                    print('Time:',time_taken)
                    running_loss = 0.0

            if not os.path.isdir("models"):
                    os.mkdir("models")
            torch.save(model.state_dict(), f'models/alexnet_cifar_{epoch+1}.pkl')    

        print('Training of AlexNet has been finished')

    elif args.phase == 'test':
        if args.weights is not None:
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(args.weights))
            else:
                model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['alexnet'], 'Pretrained_weights'))

        # Testing Accuracy
        correct = 0
        total = 0
        Net_zeros_total = 0
        Net_sizes_total = 0
        zeros_total = [0] * 10
        sizes_total = [0] * 10

        model.eval()
        with torch.no_grad():
            for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):
                image, target = data.to(device), target.to(device)
                outputs = model(image)
                
                # Sparsity calculations zone begin
                for i in range(len(outputs[1])):
                    zeros_total[i] += outputs[1][i]
                    sizes_total[i] += outputs[2][i]

                Net_zeros_total += sum(zeros_total)
                Net_sizes_total += sum(sizes_total)
                # Sparsity calculations zone end

                _, predicted = torch.max(outputs[0].data, 1) # To find the index of max-probability for each output in the BATCH
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        print()
        print('Accuracy of the network on 10000 test images: %.2f %%' % (100 * correct / total))
        print("Testing the model finished")

        # Printing Sparsity Statistics
        SR_L1   = (zeros_total[0]/sizes_total[0])
        SR_L2   = (zeros_total[1]/sizes_total[1])
        SR_L3   = (zeros_total[2]/sizes_total[2])
        SR_L4   = (zeros_total[3]/sizes_total[3])
        SR_L5   = (zeros_total[4]/sizes_total[4])
        SR_L6   = (zeros_total[5]/sizes_total[5])
        SR_L7   = (zeros_total[6]/sizes_total[6])
        SR_L8   = (zeros_total[7]/sizes_total[7])
        SR_L9   = (zeros_total[8]/sizes_total[8])
        SR_L10  = (zeros_total[9]/sizes_total[9])
        SR_Net  = (Net_zeros_total/Net_sizes_total)
        print()
        print('Sparsity rate of L1 is: %1.5f'   % (SR_L1))
        print('Sparsity rate of L2 is: %1.5f'   % (SR_L2))
        print('Sparsity rate of L3 is: %1.5f'   % (SR_L3))
        print('Sparsity rate of L4 is: %1.5f'   % (SR_L4))
        print('Sparsity rate of L5 is: %1.5f'   % (SR_L5))
        print('Sparsity rate of L6 is: %1.5f'   % (SR_L6))
        print('Sparsity rate of L7 is: %1.5f'   % (SR_L7))
        print('Sparsity rate of L8 is: %1.5f'   % (SR_L8))
        print('Sparsity rate of L9 is: %1.5f'   % (SR_L9))
        print('Sparsity rate of L10 is: %1.5f'  % (SR_L10))
        print('Sparsity rate of Net is: %1.5f'  % (SR_Net))

    elif args.phase == 'Sparsity-Attack':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        num_classes  = len(test_dataset.classes)

        c_init = 0.5
        adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total\
            = Sparsity_Attack(model, device, test_loader, num_classes, c_init, args)
        
        # Save the generated adversarial dataset to disk
        torch.save(adversarial_dataset, './adversarial_data/adversarial_dataset.pt')

        print(f"Test accuracy excluding energy attack: {initial_accuracy}")
        print(f"Test accuracy including energy attack: {final_accuracy}")
        print(f"Sparsity rate before energy attack: {sr_net_before_total}")
        print(f"Sparsity rate after energy attack: {sr_net_after_total}")
        print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
        print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")

    elif args.phase == 'profile':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()
        
        num_classes  = len(train_dataset.classes)

        # The profile function returns an array in which each element comprises a Sparsity-Map Tensor assigned to each Class
        sparsity_maps, sparsity_range, index = Profile(model, device, train_loader, num_classes, args) 
        
        # Store the Sparsity_Maps in an offline file
        print(f"We should stop profiling at data-point {index} when profiling is based on {args.method}")
        print(f"P1 = {index/(len(train_loader)-1)*100:.2f} % of training-set has been used for profiling.")

        if args.method == 'sparsity-map':
            # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_maps, "./adversarial_data/sparsity_maps.pt", pickle_protocol=4)
        
        if args.method == 'sparsity-range':
            # Save the Sparsity Range to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_range, "./adversarial_data/sparsity_ranges.pth", pickle_protocol=4)

    elif args.phase == 'Sparsity-Detect':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        # Load the sparsity-maps and sparsity-ranges from offline file (generated by profiling trainingset)
        offline_sparsity_maps = torch.load("./adversarial_data/sparsity_maps.pt")
        offline_sparsity_ranges = torch.load("./adversarial_data/sparsity_ranges.pth")
        
        num_classes  = len(offline_sparsity_maps) # or length of (offline_sparsity_ranges)

        # Prints the number of detected adversarials in each class
        Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes)

# Train:   python3 main.py --phase train
# Test:    python3 main.py --phase test --weights models/alexnet_cifar_95_0.8813.pkl
# Attack:  python3 main.py --phase Sparsity-Attack --store_attack --constrained --imax 100 --beta 15 --eps 0.99 --eps_iter 0.01 --weights models/alexnet_cifar_95_0.8813.pkl
# Profile: python3 main.py --phase profile --method sparsity-map/range --weights models/alexnet_cifar_95_0.8813.pkl
# Detect:  python3 main.py --phase Sparsity-Detect --weights models/alexnet_cifar_95_0.8813.pkl
