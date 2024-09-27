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

import datetime

from models.source_googlenet import googlenet_stat
from models.googlenet import googlenet

import torchvision.utils as vutils
import random
   
#####################################constants###########################
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

MILESTONES = [60, 120, 160]
#########################################################################


class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]              

########################################################################################

        
   
        
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

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]

def clip_tensor(input_tensor, eps_in, batch_size, input_tensor_clean, min_in, max_in):

    with torch.no_grad():
        clapped_in = input_tensor

        for i in range(batch_size):
            torch.clamp(clapped_in[i], min=(min_in[i]-eps_in), max=(max_in[i]+eps_in), out=clapped_in[i])

        
        torch.clamp(clapped_in[:,0, :,:], min=((0-0.5070751592371323)/0.2673342858792401), max=((1-0.5070751592371323)/0.2673342858792401) , out=clapped_in[:,0, :,:])
        torch.clamp(clapped_in[:,1, :,:], min=((0-0.48654887331495095)/0.2564384629170883), max=((1-0.48654887331495095)/0.2564384629170883), out=clapped_in[:,1, :,:])
        torch.clamp(clapped_in[:,2, :,:], min=((0-0.4409178433670343)/0.27615047132568404), max=((1-0.4409178433670343)/0.27615047132568404), out=clapped_in[:,2, :,:])
    

    return clapped_in


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

def criterion_sparsity(y_pred, y):
    return torch.mean((y_pred - y) ** 2)






# Convert a clean image into an advesarial image
#bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
def convert_clean_to_adversarial(model_active_in, device, inputs_dirty, x_clean, init_pred, c_init, args):

    c_min = 0
    c_max = 1
    coeff = torch.full((args.batch_size,), c_init, device=device)

    max_clean, _ = torch.max(x_clean.view(args.batch_size, -1), dim=1)
    min_clean, _ = torch.min(x_clean.view(args.batch_size, -1), dim=1)

    x = inputs_dirty.clone().detach().requires_grad_(True)
    #x.requires_grad = True
    optimizer = optim.Adam([x], lr=args.lr, amsgrad=True) #, weight_decay=1e-4)

    image_loss = nn.MSELoss()
        
    for epoch in range(args.imax):  # loop over the dataset multiple times     
            
        # zero the parameter gradients
        optimizer.zero_grad()

            
        # forward + backward + optimize
        outputs = model_active_in(x)

        final_pred = outputs[0].max(1, keepdim=False)[1]
        coeff = torch.stack([torch.where(init_pred[i] == final_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
 
           
        loss1 = torch.stack([F.cross_entropy(outputs[0][j].unsqueeze(0), init_pred[j].unsqueeze(0)) for j in range(args.batch_size)])


        criteria_sparse_loss = nn.MSELoss()#reduction=‘mean’)

        perfectt = torch.zeros(args.batch_size, 1, requires_grad=False).to(device)
        for i in range(args.batch_size):
            perfectt[i, 0] = torch.tensor([1.], requires_grad=False)

        for j in range(args.batch_size):
            assert(outputs[1][j].unsqueeze(0) < 974592)
                
        loss2 = torch.stack([criteria_sparse_loss(outputs[1][j].unsqueeze(0)/974592, perfectt[j].unsqueeze(0)) for j in range(args.batch_size)])

        #####################################
        loss3 = torch.stack([image_loss(x[j], x_clean[j]) for j in range(args.batch_size)])

        #######################################
        
        #net_sizes = 974592 = model2_in.size_of_my_fc_GoogleNet_stat()
        #490112 = 64*55*55 + 192*27*27 + 384*13*13 + 256*13*13 + 256*13*13 + 4096 + 1024
            
        loss = coeff*loss1 + loss2 + loss3

        loss.backward(torch.ones_like(loss))
        optimizer.step()

        if args.constrained:
            x = clip_tensor(x, args.eps, args.batch_size, x_clean, min_clean, max_clean)

       
        # print statistics
        #running_loss += loss.item()
        
    return x




# Generates sparsity attack for P2=50% of each class in test_dataset
#aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
def Sparsity_Attack_Generation(model_active_in, model2_in, device, test_loader_clean_in, test_loader_dirty_in, num_classes, c_init, args):

    print('started at:', datetime.datetime.now() )
    print(" ")
    model_active_in.eval()
    model2_in.eval()




    ################
    num_processed_images = 0
    correct_before = 0
    correct_after = 0
    l2_norms = []
    total_net_ones_before = 0
    total_net_sizes_before = 0
    total_net_ones_after = 0
    total_net_sizes_after = 0

    adversarial_data = []    

    #net_sizes = 974592 = model2_in.size_of_my_fc_GoogleNet_stat()
    net_sizes = 974592*args.batch_size

    count = 0

   
    
    for ((data_clean, target_clean) , (data_dirty, target_dirty)) in zip(test_loader_clean, test_loader_dirty):
        inputs_clean, target_clean = data_clean.to(device), target_clean.to(device)
        inputs_dirty, target_dirty = data_dirty.to(device), target_dirty.to(device)


        # To inject the Clean Image to model to compute the accuracy and sparsity rate
        output_init = model2_in(inputs_clean)
        net_ones = output_init[1]
        #net_sizes = (64*55*55 + 192*27*27 + 384*13*13 + 256*13*13 + 256*13*13 + 4096 + 1024)*args.batch_size

        
        total_net_ones_before += net_ones
        total_net_sizes_before += net_sizes
        init_pred = output_init[0].max(1, keepdim=False)[1]
        correct_before += (init_pred == target_clean).sum().item()
   
        
        count = count +1

        if((count%200)==0):
            print("count=", count)
            print(datetime.datetime.now() )
            #break
     

        ############################

        
        x_new =convert_clean_to_adversarial(model_active_in, device, inputs_dirty, inputs_clean, init_pred, c_init, args)

        
        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of 100% for each class
        if args.store_attack:
            #adversarial_data.append((x_new, target), 1))
            adversarial_data.append( (x_new, target_clean) )


        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-inputs_clean).view(args.batch_size, -1), p=2, dim=1)
        l2norm_diff = l2norm_diff/ ( (x_new.view(args.batch_size, -1)).size(1) )
        
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

     

        # To inject the Adversarial Image to model to compute the accuracy and sparsity rate
        output_new = model2_in(x_new)
        final_pred = output_new[0].max(1, keepdim=False)[1]
        correct_after += (final_pred == target_clean).sum().item()

        # Re-compute the sparsity rate using the perturbed input
        net_ones = output_new[1]
        net_sizes = 974592*args.batch_size
        #net_sizes = (64*55*55 + 192*27*27 + 384*13*13 + 256*13*13 + 256*13*13 + 4096 + 1024)*args.batch_size
        total_net_ones_after += net_ones
        total_net_sizes_after += net_sizes
        
        num_processed_images += args.batch_size


        if args.save_samples:
            img_grid_clean = torchvision.utils.make_grid(inputs_clean)
            vutils.save_image(img_grid_clean.detach(),
                              './sample_images/clean_idx_%03d.png' % (count),
                              normalize=True)
      
            img_grid_dirty = torchvision.utils.make_grid(x_new)
            vutils.save_image(img_grid_dirty.detach(),
                              './sample_images/dirty_idx_%03d.png' % (count),
                              normalize=True)

        

    # To Create a new dataset using the AdversarialDataset class
    if args.store_attack:
        adversarial_dataset = AdversarialDataset(adversarial_data)
    else:
        adversarial_dataset = 0
        

    # Calculate overal accuracy of all test data after sparsity attack
    first_acc = correct_before/float(num_processed_images)
    final_acc = correct_after/float(num_processed_images)

    print(" ")
    print('ended at:', datetime.datetime.now() )
    print(" ")

    return adversarial_dataset, first_acc, final_acc, l2_norms, (1-(total_net_ones_before/total_net_sizes_before)), (1-(total_net_ones_after/total_net_sizes_after))

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
    parser.add_argument('--batch_size', default=20, type=int, help="Batch size of 4 is a good choice for Training")
    parser.add_argument('--phase', default='sparsity-attack', help="train, test, profile, sparsity-attack, sparsity-detect")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism based on sparsity-map or sparsity-range")
    parser.add_argument('--weights', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/cifar100/2_copy_weight/googlenet_cifar100_fc_one_out.pkl', help="The path to the saved weights")
    parser.add_argument('--dataset', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/cifar100/cifar100_dataset', help="The path to the train and test datasets")
    parser.add_argument('--beta', default=50, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--constrained', action='store_true', help="To enable clipping the generated purturbed data")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN architecture") # power
    parser.add_argument('--GAN', action='store_true', help="To test the GAN dataset in the test phase") # GAN


    parser.add_argument('--save_samples', action='store_true', help="To save sample adversarial images")
    parser.add_argument('--store_attack', action='store_true', help="To enable or disable algorithm to store generated adversarials in an output dataset")

    parser.add_argument('--im_index_fisrt', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=10000, type=int, help="The last index of the dataset")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")    
    parser.add_argument('--cont_adv_geneartion', action='store_true', help="Reads advesarial images from previous imax-loop and contnues optmizing the images")
    parser.add_argument('--manualSeed', type=int,  default=13, help='manual seed')

    
    args = parser.parse_args()
    print(args)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)  


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    
    

    if args.phase == 'sparsity-attack':

        test_dataset_clean  = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=transform)
        test_dataset_sub_clean = torch.utils.data.Subset(test_dataset_clean,  list(range(args.im_index_fisrt, args.im_index_last)))        
        test_loader_clean = DataLoader(test_dataset_sub_clean, shuffle=False, num_workers=1, batch_size=args.batch_size)
 
        
        if args.cont_adv_geneartion:
            test_loader_dirty = torch.load('./adversarial_dataset_c100_in.pt', map_location=torch.device('cuda'))
        else:
            test_dataset_dirty  = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=transform)
            test_dataset_sub_dirty = torch.utils.data.Subset(test_dataset_dirty,  list(range(args.im_index_fisrt, args.im_index_last)))        
            test_loader_dirty = DataLoader(test_dataset_sub_dirty, shuffle=False, num_workers=1, batch_size=args.batch_size)
              
        
            #test_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform)
            #test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.phase == 'profile':
        train_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
    elif args.phase == 'sparsity-detect':
        # To load the adversarial_dataset from the offline file generated by Sparsity-Attack function (50% are polluted)
        test_dataset = torch.load('./adversarial_data/adversarial_dataset_unconstrained.pt', map_location=torch.device('cpu'))
    elif args.GAN:
        # To load the GAN/EA dataset from the offline file generated by GAN/EA algorithm # GAN
        # test_dataset = torch.load('./adversarial_data/Reconstruct_sparasity_constrained_L5000_R100_lr50.pt', map_location=torch.device('cpu')) # GAN : Testset
        test_dataset = torch.load('./adversarial_data/adversarial_dataset_constrained.pt', map_location=torch.device('cpu')) # Energy Attack: Testset
        test_loader  = DataLoader(test_dataset, batch_size=args.batch_size)
    else:
        train_data = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)



    # To use the genuine AlexNet with 1000 classes as the Network Model
    model_active = googlenet(beta=args.beta)
    model2 = googlenet_stat(beta=args.beta)
    #print(model_active)
 

    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_active.to(device)
    model2.to(device)
    print(f"\n A {device} is assigned for processing! \n")




    if args.save_samples:
        if not os.path.isdir('sample_images'):
            os.mkdir('sample_images')
 
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9)

    if args.phase == 'train':
        # Training
        EPOCHS = args.epochs
        for epoch in range(EPOCHS):
            model2.train()
            running_loss = 0.0
            start_time = time.time()
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                output = model2(inputs)
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
            torch.save(model2.state_dict(), f'models/alexnet_cifar_{epoch+1}.pkl')    

        print('Training of AlexNet has been finished')

    elif args.phase == 'test':
        if args.weights is not None:
            if torch.cuda.is_available():
                model2.load_state_dict(torch.load(args.weights))
            else:
                model2.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
        else:
            print("weight is not found")
            exit(0)

        # Testing Accuracy
        correct = 0
        total = 0
        net_zeros_total = 0
        net_sizes_total = 0
        zeros_total = [0] * 10
        sizes_total = [0] * 10

        model2.eval()
        with torch.no_grad():
            for index, (data, target) in enumerate(tqdm(test_loader, desc='Data Progress')):
                image, target = data.to(device), target.to(device)
                outputs = model2(image)
                
                # Sparsity calculations zone begin
                for i in range(len(outputs[1])):
                    zeros_total[i] += outputs[1][i]
                    sizes_total[i] += outputs[2][i]

                net_zeros_total += sum(zeros_total)
                net_sizes_total += sum(sizes_total)
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
        SR_Net  = (net_zeros_total/net_sizes_total)
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

    elif args.phase == 'sparsity-attack':
        if args.weights is not None:
            model_active.load_state_dict(torch.load(args.weights, map_location=device))
            model2.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        #model_active.train()
        #model2.eval()

        num_classes  = 100 #len(test_dataset.classes)

        c_init = 0.5
        adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total = Sparsity_Attack_Generation(model_active, model2, device, test_loader_clean, test_loader_dirty,num_classes, c_init, args)
     


        DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
        #time of we run the script
        TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)

        image_out_name = 'adversarial_dataset_c100'
        image_out_name = os.path.join(image_out_name, '_lr_', str(args.lr), '_', TIME_NOW, '.pt')
        image_out_name = image_out_name.replace('/',  '')
        
        
        # Save the generated adversarial dataset to disk
        if args.store_attack:
            torch.save(adversarial_dataset, image_out_name) #'./adversarial_dataset.pt')

        print(f"Test accuracy excluding energy attack: {initial_accuracy}")
        print(f"Test accuracy including energy attack: {final_accuracy}")
        print(f"Sparsity rate before energy attack: {sr_net_before_total}")
        print(f"Sparsity rate after energy attack: {sr_net_after_total}")
        print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
        print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")

    elif args.phase == 'profile':
        if args.weights is not None:
            model2.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model2.eval()
        
        num_classes  = 100 #len(train_dataset.classes)

        # The profile function returns an array in which each element comprises a Sparsity-Map Tensor assigned to each Class
        sparsity_maps, sparsity_range, index = Profile(model2, device, train_loader, num_classes, args) 
        
        # Store the Sparsity_Maps in an offline file
        print(f"We should stop profiling at data-point {index} when profiling is based on {args.method}")
        print(f"P1 = {index/(len(train_loader)-1)*100:.2f} % of training-set has been used for profiling.")

        if args.method == 'sparsity-map':
            # Save the Sparsity Maps to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_maps, "./adversarial_data/sparsity_maps.pt", pickle_protocol=4)
        
        if args.method == 'sparsity-range':
            # Save the Sparsity Range to a compressed file using pickle protocol 4 (for Python 3.4+)
            torch.save(sparsity_range, "./adversarial_data/sparsity_ranges.pth", pickle_protocol=4)

    elif args.phase == 'sparsity-detect':
        if args.weights is not None:
            model2.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model2.eval()

        # Load the sparsity-maps and sparsity-ranges from offline file (generated by profiling trainingset)
        offline_sparsity_maps = torch.load("./adversarial_data/sparsity_maps.pt")
        offline_sparsity_ranges = torch.load("./adversarial_data/sparsity_ranges.pth")
        
        num_classes  = len(offline_sparsity_maps) # or length of (offline_sparsity_ranges)

        # Prints the number of detected adversarials in each class
        Sparsity_Attack_Detection(model2, device, offline_sparsity_maps, offline_sparsity_ranges, test_dataset, num_classes)

# Attack:  python3 main.py --phase sparsity-attack --imax 100 --beta 15 --eps 0.99 --batch_size 10 --store_attack --constrained --dataset data --weights models/alexnet_cifar_95_0.8813.pkl
