import torch
import argparse
import os
import sys
import numpy as np
from torch import nn
from torch.nn import Module



from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torchvision.datasets as dset
import random


class LeNet5(nn.Module):
    def __init__(self, beta=15):
        super(LeNet5, self).__init__()
        self.beta  = beta
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

      
        
    def forward(self, x):
                       
        x = self.conv1(x)
        x = self.relu1(x)
             
        x = self.pool1(x)     
        x = self.conv2(x)
        x = self.relu2(x)
      
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
      
        x = self.fc1(x)
        x = self.relu3(x)
     
        x = self.fc2(x)
        x = self.relu4(x)
    
        x = self.fc3(x)
        x = self.relu5(x)
          
        return x





# To compute Sparsity-Map for each layer's output
def sparsity_map(input_tensor):
    # To create a tensor with all zeros and with the same shape of input tensor
    maps = torch.zeros_like(input_tensor, dtype=torch.uint8)
    # To find the positions where the input tensor is zero
    zero_positions = torch.eq(input_tensor, 0)
    # To set the corresponding positions in the bit_map to 1
    maps = maps.masked_fill(zero_positions, 1)
    return maps

# FGSM attack function to create adversarial example (perturbed image)
def fgsm_attack(image_clean, epsilon, data_grad):
    """
    data_grad is the gradient of the loss w.r.t the input image
    """
    sign_data_grad  = data_grad.sign() # Collect the element-wise sign of the data gradient
    perturbed_image = image_clean + epsilon * sign_data_grad # Creates the perturbed image by adjusting each pixel of the input image
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # Clipping perturbed_image within [0,1] range
    return perturbed_image
    
# For each sample in test set, this computes the gradient of the loss w.r.t the input data (data_grad), creates a perturbed image with fgsm_attack (perturbed_data)
# Then checks to see if the perturbed example is adversarial. To test the model accuracy, it saves and returns some successful adversarial examples to be visualized later
def FGSM_Test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor to True to compute gradients with respect to input data
        data.requires_grad = True

        output = model(data)
        init_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability (mohammad added [0])

        # If the initial prediction (before attack) is wrong, don't bother FGSM to generate adversarial
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output[0], target) # mohammad added [0]

        # Zero all existing gradients and Calculate gradients of model in backward
        model.zero_grad()
        loss.backward()

        # Collect 'data gradients'
        data_grad = data.grad.data

        # Call fgsm_attack function to generate perturbed data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Final prediction after applying attack
        final_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability (mohammad added [0])

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving epsilon=0 examples (attack free)
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # In the case of misprediction, It saves some adv-examples for later visualization
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for the given epsilon after adversarial attack
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# To calculate range
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

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]
def clip_tensor(input_tensor, eps, batch_size):
    
    tensor_norm = torch.stack([torch.norm(input_tensor[i], p=2) for i in range(batch_size)])

    # If the L2 norm of generated adversarial is greater than the eps, we scale it down by a factor of (max_norm / tensor_norm)
    clipped_tensor = torch.stack([torch.where(tensor_norm[i] > eps, input_tensor[i] * (eps / tensor_norm[i]), input_tensor[i]) for i in range(batch_size)])

    max_element, _ = torch.max(clipped_tensor.view(batch_size, -1), dim=1)
    
    # If the maximum element is larger than the 1, we again scale it down by a factor of 1/max_element
    clipped_tensor = torch.stack([torch.where(max_element[i] > 1, clipped_tensor[i].div_(max_element[i]), clipped_tensor[i]) for i in range(batch_size)])
    
    # The default to implement the constrained mode
    # return torch.clamp(clipped_tensor, 0, 1)
    return clipped_tensor
    
# Generates sparsity attack for 50% of images in each class of original testset
def Sparsity_Attack_Generation(model, device, test_loader, num_classes, c_init, args):
    
    num_processed_images = 0
    correct_before = 0
    correct_after = 0
    l2_norms = []
    total_net_zeros_before = 0
    total_net_sizes_before = 0
    total_net_zeros_after = 0
    total_net_sizes_after = 0

    adversarial_data = []
    # A value of 1 means the sparsity attack has polluted the last processed input data of the same class, so we need to leave the current input clean.
    # A value of 0 means the sparsity attack has left the last processed input data of the same class clean, so we need to pollute the current input.
    last_status_of_class  = [0] * num_classes
    num_of_items_in_class = [0] * num_classes
    num_of_adver_in_class = [0] * num_classes

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    #for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):
    for (data, target) in test_loader:

        data, target = data.to(device), target.to(device)
        i_max = args.imax
        c_min = 0
        c_max = 1
        momentum = 0.9
        coeff = torch.full((args.batch_size,), c_init, device=device)
        eps = args.eps
        eps_iter = args.eps_iter

        # To inject the Clean Image to model to compute the accuracy and sparsity rate
        output = model(data, True)
        Net_zeros = sum(output[1])
        Net_sizes = sum(output[2])
        total_net_zeros_before += Net_zeros
        total_net_sizes_before += Net_sizes
        init_pred = output[0].max(1, keepdim=False)[1]
        correct_before += (init_pred == target).sum().item()

        x = data
        g = torch.zeros_like(x)
        
        for i in range(i_max):

            # To set requires_grad attribute of X to True to compute the gradient of loss function (l_x) w.r.t input X
            x = x.clone().detach().requires_grad_(True)
            
            output = model(x, False)
            
            if (i>0):
                final_pred = output[0].max(1, keepdim=False)[1]
                coeff = torch.stack([torch.where(init_pred[i] == final_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
            
            # To compute cross-entropy loss independently for each image in the batch
            l_ce = torch.stack([F.cross_entropy(output[0][j].unsqueeze(0), init_pred[j].unsqueeze(0)) for j in range(args.batch_size)])
            
            # To compute l_sparsity (using Tanh function) independently for each image in the batch
            l_sparsity = output[3]
            
            l_x = l_sparsity + (coeff * l_ce)
                    
            optimizer.zero_grad()
            l_x.backward(torch.ones_like(l_x)) # Using torch.ones_like(l_x) ensures that the gradients of each element in l_x is computed independently 
                
            g = (momentum * g) + x.grad.data
            x_new = torch.stack([x[k] - (eps_iter * (g[k] / torch.norm(g[k], p=2))) for k in range(args.batch_size)])
            if args.constrained:
                x_new = clip_tensor(x_new, eps, args.batch_size)
            x = x_new

        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of 100% for each class
        if args.store_attack:
            adversarial_data.append((x_new, target, 1))
            
        """
        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of 50% for each class
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
        """

        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-data).view(args.batch_size, -1), p=2, dim=1)
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

        # To inject the Adversarial Image to model to compute the accuracy and sparsity rate
        output = model(x_new, True)
        final_pred = output[0].max(1, keepdim=False)[1]
        correct_after += (final_pred == target).sum().item()

        # Re-compute the sparsity rate using the perturbed input
        Net_zeros = sum(output[1])
        Net_sizes = sum(output[2])
        total_net_zeros_after += Net_zeros
        total_net_sizes_after += Net_sizes

        num_processed_images += args.batch_size

    # To Create a new dataset using the AdversarialDataset class
    if args.store_attack:
        adversarial_dataset = AdversarialDataset(adversarial_data)
    else:
        adversarial_dataset = 0
        
        print(f"Distribution of data for each class: {num_of_items_in_class}")
    print(f"Number of generated adversarials in each class: {num_of_adver_in_class}")

    # Calculate overall accuracy of all test data after sparsity attack
    first_acc = correct_before/float((num_processed_images))
    final_acc = correct_after/float((num_processed_images))

    return adversarial_dataset, first_acc, final_acc, l2_norms, (total_net_zeros_before/total_net_sizes_before), (total_net_zeros_after/total_net_sizes_after)

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

        pred = output[0].max(1, keepdim=True)[1]
        if pred.item() == target.item():
            if args.method == 'sparsity-range':
                current_sparsity_rate = [0.0] * num_layers  # Initialize the current_sparsity list reading each data input
                for i in range(5):
                    numerator_index = 2 * i + 1
                    denominator_index = 2 * i + 2
                    current_sparsity_rate[i] = output[numerator_index] / output[denominator_index]
                range_for_each_class = calculate_range(current_sparsity_rate, range_for_each_class, target.item())

            elif args.method == 'sparsity-map':
                if updated_maps[target.item()] == None:
                    updated_maps[target.item()] = output[16]
                    prev_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item() # Computes the number of 1s in the updated_maps
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()])
                else:
                    updated_maps [target.item()] = torch.bitwise_or(updated_maps[target.item()], output[16]) # output[12] is the Sparsity-Map associated with the current input
                    curr_num_ones[target.item()] = torch.sum(updated_maps[target.item()]).item()
                    diff_num_ones[target.item()] = abs(curr_num_ones[target.item()] - prev_num_ones[target.item()]) # Computes the difference between the number of 1s in the current and previous Sparsity-Maps
                    prev_num_ones[target.item()] = curr_num_ones[target.item()]

                if all(num < 1 for num in diff_num_ones):
                    print(diff_num_ones)
                    break

    return updated_maps, range_for_each_class, index

def Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes):
    
    # Initialization for sparsity-maps
    sim_threshold = 0.323
    correctly_predicted_adversarial_for_class_map = [0] * num_classes
    generated_adversarial_for_class = [0] * num_classes 
    num_of_items_in_class = [0] * num_classes
    correctly_predicted_adversarial_ratio_map = []

    # Initialization for sparsity-ranges
    num_layers = 7
    layer_inclusion_threshold = num_layers - 6
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
            numerator_index = 2 * L + 1
            denominator_index = 2 * L + 2
            current_sparsity_rate[L] = output[numerator_index] / output[denominator_index]
        
        in_range_status = [0] * num_layers
        for M in range(num_layers):
            if not offline_sparsity_ranges[pred.item()][M][0] <= current_sparsity_rate[M] <= offline_sparsity_ranges[pred.item()][M][1]:
                in_range_status[M] = 1
        
        if sum(in_range_status) >= layer_inclusion_threshold:
            if adversarial == 1:
                correctly_predicted_adversarial_for_class_range[target.item()] += 1

        ############################################## sparsity-map #######################################################################
        
        sim_rate = jaccard_similarity(offline_sparsity_maps[pred.item()], output[16])

        # if the following condition is True, we predict that the input is adversarial
        if sim_rate <= sim_threshold:
            if adversarial == 1:
                correctly_predicted_adversarial_for_class_map[target.item()] += 1

        # To check the real adversarial status for the same predicted class
        if adversarial == 1:
            generated_adversarial_for_class[target.item()] += 1 

        num_of_items_in_class[target.item()] += 1

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
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help="Initial learning rate")
    parser.add_argument('--phase', default='train', help="train, test, profile, FGSM, sparsity-attack, sparsity-detect")
    parser.add_argument('--method', default='sparsity-map', help="profiling can be performed based on sparsity-map or sparsity-range")
    parser.add_argument('--weights', default=None, help="The path to the saved weights. Should be specified when testing")
    parser.add_argument('--dataset', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/MNIST/mnist_dataset', help="The path to the train and test datasets")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--omax', default=5, type=int, help="Maximum iterations in the outer loop of Sparsity Attack function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon in each inner loop iteration")
    parser.add_argument('--constrained', action='store_true', help="To avoid clipping the generated purturbed data")
    parser.add_argument('--store_attack', action='store_true', help="To lead the algorithm to store the generated adversarials")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN architecture") # power
    parser.add_argument('--GAN', action='store_true', help="To test the GAN dataset in the test phase") # GAN

    parser.add_argument('--manualSeed', type=int,  default=13, help='manual seed')

    args = parser.parse_args()
    print(f"{args}\n")

    # MNIST dataset and dataloader declaration
    if args.phase == 'FGSM':
        test_dataset = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=True)
    elif args.phase == 'sparsity-attack':
        test_dataset = mnist.MNIST(root=args.dataset, train=False, download=True, transform=transforms.ToTensor())
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.phase == 'sparsity-detect':
        # To load the adversarial_dataset from the offline file generated by sparsity-attack function (50% are polluted)
        test_dataset = torch.load('./adversarial_data/adversarial_dataset_constrained.pt', map_location=torch.device('cpu'))
    elif args.phase == 'profile':
        train_dataset = mnist.MNIST(root=args.dataset, train=True, download=True, transform=transforms.ToTensor())
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
    elif args.GAN:
        # To load the GAN/EA dataset from the offline file generated by GAN/EA algorithm # GAN
        # test_dataset = torch.load('./adversarial_data/Reconstruct_sparasity_constrained_L5000_R100_lr50.pt', map_location=torch.device('cpu')) # GAN : Testset
        test_dataset = torch.load('./adversarial_data/adversarial_dataset_constrained.pt', map_location=torch.device('cpu')) # Energy Attack: Testset
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)
    else:
        test_dataset  = dset.MNIST(root=args.dataset, train=False, transform=transforms.ToTensor()) # Original Testset

        #test_dataset = dset.MNIST(root=args.dataset, download=True, train=False,
        #                          transform=transforms.Compose([
        #                              transforms.Resize(28),
        #                              transforms.ToTensor(),
        #                              transforms.Normalize((0.5,), (0.5,)),
        #                          ]))

        
        test_loader   = DataLoader(test_dataset, batch_size=args.batch_size)

        train_dataset  = dset.MNIST(root=args.dataset, train=True, transform=transforms.ToTensor()) # Original Testset

        #train_dataset = dset.MNIST(root=args.dataset, download=True, train=True,
        #                          transform=transforms.Compose([
        #                              transforms.Resize(28),
        #                              transforms.ToTensor(),
        #                              transforms.Normalize((0.5,), (0.5,)),
        #                          ]))
        
        train_loader  = DataLoader(train_dataset, batch_size=args.batch_size)



    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
        
    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n A {device} is assigned for processing! \n")

    # Network Initialization
    model = LeNet5().to(device)
    print(f"\n{model}\n")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = CrossEntropyLoss()
    
    if args.phase == 'train':
        EPOCHS = args.epochs
        prev_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            for idx, (train_x, train_label) in enumerate(train_loader):
                optimizer.zero_grad()
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                preds = model(train_x.float())
                loss = loss_fn(preds, train_label.long()) # mohammad added: [0]
                loss.backward()
                optimizer.step()
    
            correct_num = 0
            sample_num = 0
            model.eval()
            for idx, (image, label) in enumerate(test_loader):
                image = image.to(device)
                label = label.to(device)
                preds = model(image.float().detach())
                preds =torch.argmax(preds, dim=-1) # mohammad added [0]
                current_correct_num = preds == label
                correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                sample_num += current_correct_num.shape[0]
            acc = correct_num / sample_num
            print('Accuracy in Epoch %3d: %1.4f' % (epoch+1, acc*100), flush=True)
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model.state_dict(), 'models/mnist_{:.4f}.pkl'.format(acc))
            if np.abs(acc - prev_acc) < 1e-4:
                break
            prev_acc = acc
        print("Training the model has been finished")

    elif args.phase == 'test':
        if args.weights is not None:
            # Load the pretrained model
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()

        correct_num = 0
        sample_num = 0
        
     

        # Set the model in evaluation mode
        model.eval()
        # for (index, input) in enumerate(tqdm(test_loader, desc='Data Progress')): # To examine GAN dataset
        # for index, (image, label, adversarial) in enumerate(tqdm(test_dataset, desc='Data Progress')): # To examine Energy Attack dataset
        for index, (image, label) in enumerate(tqdm(test_loader, desc='Data Progress')): # To examine Original test dataset
            image = image.to(device)
            label = label.to(device)
            #image = input[0].to(device) # GAN
            #label = input[1].to(device) # GAN
            output = model(image, True)
                   
          
            
            preds =torch.argmax(output, dim=-1)
            current_correct_num = preds == label
            correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            sample_num += current_correct_num.shape[0]

        # To print accuracy statistics        
        acc = correct_num / sample_num
        print()
        print('Accuracy of Testing is (percent): %1.2f' % (acc*100), flush=True)
        
      
        
    elif args.phase == 'FGSM':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()

        # FGSM Initialization
        epsilons = [0, 0.007, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        # Set the model in evaluation mode
        model.eval()
        accuracies = []
        examples = []

        # Run test for each epsilon
        for eps in epsilons:
            acc, ex = FGSM_Test(model, device, test_loader, eps)
            accuracies.append(acc)
            examples.append(ex)

        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
    
    elif args.phase == 'sparsity-attack':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        num_classes  = len(test_dataset.classes)

        c_init = 0.5
        adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total = Sparsity_Attack_Generation(model, device, test_loader, num_classes, c_init, args)
        
        # Save the generated adversarial dataset to disk

        if args.store_attack:
            torch.save(adversarial_dataset, './adversarial_dataset.pt')

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
        print(f"P1 = {index/(len(train_loader)-1)*100} % of training-set has been used for profiling.")

        if args.method == 'sparsity-map':
            # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_maps, "./adversarial_data/sparsity_maps.pt", pickle_protocol=4)
        
        if args.method == 'sparsity-range':
            # Save the Sparsity Range to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_range, "./adversarial_data/sparsity_ranges.pth", pickle_protocol=4)

    elif args.phase == 'sparsity-detect':
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        # Load the sparsity-maps and sparsity-ranges from offline file (generated by profiling training set)
        offline_sparsity_maps = torch.load("./adversarial_data/sparsity_maps.pt")
        offline_sparsity_ranges = torch.load("./adversarial_data/sparsity_ranges.pth")
        
        num_classes  = len(offline_sparsity_maps) # or length of (offline_sparsity_ranges)

        # Prints the number of detected adversarials in each class
        Sparsity_Attack_Detection(model, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes)

# Training:         python3 main.py --phase train
# Testing:          python3 main.py --phase test --batch_size 4 --power --GAN --weights models/mnist_0.9876.pkl
# FGSM:             python3 main.py --phase FGSM --weights models/mnist_0.9876.pkl
# Profiling:        python3 main.py --phase profile --method sparsity-map/range --weights models/mnist_0.9876.pkl
# Attack Execution: python3 main.py --phase sparsity-attack --eps 0.99 --eps_iter 0.04 --imax 200 --beta 31 --store_attack --constrained --weights models/mnist_0.9876.pkl
# Attack Detection: python3 main.py --phase sparsity-detect --weights models/mnist_0.9876.pkl
