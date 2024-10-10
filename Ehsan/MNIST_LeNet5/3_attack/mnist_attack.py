import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import datetime
import random
import os
import sys
import argparse
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from models.source_mnist_model import lenet5
from models.mnist_model import lenet5_active

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]            
        
def sparsity_rate(input_tensor):
    zeros = torch.count_nonzero(torch.eq(input_tensor, 0)).item()
    activation_count = input_tensor.numel()
    return zeros, activation_count

# To compute tanh with beta parameter
def tanh(input_tensor, beta):
    output = torch.tanh(beta * input_tensor)
    return output

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]
def clip_tensor(input_tensor, eps_in, batch_size, min_in, max_in):

    with torch.no_grad():
        clapped_in = input_tensor

        for i in range(batch_size):
            torch.clamp(clapped_in[i], min=(min_in[i]-eps_in), max=(max_in[i]+eps_in), out=clapped_in[i])
        
        torch.clamp(clapped_in, min=0, max=1, out=clapped_in)

    return clapped_in

# Convert a clean image into an advesarial image
def convert_clean_to_adversarial(model1_active_in, device, inputs_dirty, inputs_clean, init_pred, c_init, args):

    c_min = 0
    c_max = 1
    coeff = torch.full((args.batch_size,), c_init, device=device)

    max_clean, _ = torch.max(inputs_clean.view(args.batch_size, -1), dim=1)
    min_clean, _ = torch.min(inputs_clean.view(args.batch_size, -1), dim=1)

    x = inputs_dirty.clone().detach().requires_grad_(True)
    
    """optimizer = optim.SGD([x], lr=args.lr, momentum=0.9)"""
    optimizer = optim.Adam([x], lr=args.lr, amsgrad=True) # , betas=(0.99, 0.999), eps=1e-20, weight_decay=1e-4)
    image_loss = nn.MSELoss()
    sparsity_loss = nn.MSELoss()
        
    for epoch in range(args.imax):

        # Initialize the x gradients with zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model1_active_in(x)

        final_pred = outputs[0].max(1, keepdim=False)[1]
        coeff = torch.stack([torch.where(init_pred[i] == final_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
           
        loss1 = torch.stack([F.cross_entropy(outputs[0][j].unsqueeze(0), init_pred[j].unsqueeze(0)) for j in range(args.batch_size)])

        perfectt = torch.ones(args.batch_size, 1, requires_grad=False).to(device)

        for j in range(args.batch_size):
            assert(outputs[1][j].unsqueeze(0) < 2108)
                
        loss2 = torch.stack([sparsity_loss(outputs[1][j].unsqueeze(0)/2108, perfectt[j].unsqueeze(0)) for j in range(args.batch_size)])

        loss3 = torch.stack([image_loss(x[j], inputs_clean[j]) for j in range(args.batch_size)])

        loss = coeff * loss1 + loss2 + loss3

        loss.backward(torch.ones_like(loss))

        optimizer.step()
        
        if args.constrained:
            x = clip_tensor(x, args.eps, args.batch_size, min_clean, max_clean)
 
    return x

# Generates sparsity attack for all images in test_dataset
def Sparsity_Attack_Generation(model1_active_in, model2_in, device, test_loader_clean_in, test_loader_dirty_in, num_classes, c_init, args):

    print('\nStarted at: ', datetime.datetime.now(), '\n')
    
    model1_active_in.eval()
    model2_in.eval()

    num_processed_images = 0
    correct_before = 0
    correct_after = 0
    l2_norms = []
    total_net_ones_before = 0
    total_net_sizes_before = 0
    total_net_ones_after = 0
    total_net_sizes_after = 0

    adversarial_data = []    

    net_sizes = 2108 * args.batch_size
    # net_sizes = 2108 = 1*28*28 + 6*12*12 + 256 + 120 + 84

    count = 0

    for ((data_clean, target_clean) , (data_dirty, target_dirty)) in zip(test_loader_clean_in, test_loader_dirty_in):
        
        inputs_clean, target_clean = data_clean.to(device), target_clean.to(device)
        inputs_dirty, target_dirty = data_dirty.to(device), target_dirty.to(device)

        # To inject the clean image into "base model" and compute the "number of non-zeros or net_ones" and "accuracy"
        output_init = model2_in(inputs_clean)
        net_ones = output_init[1]
        total_net_ones_before += net_ones
        total_net_sizes_before += net_sizes
        init_pred = output_init[0].max(dim=1, keepdim=False)[1]
        correct_before += (init_pred == target_clean).sum().item()

        count = count + 1

        if count%200==0 :
            print("\ncount=", count)
            print(datetime.datetime.now(), '\n')

        x_new = convert_clean_to_adversarial (model1_active_in, device, inputs_dirty, inputs_clean, init_pred, c_init, args)

        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of 100% for each class
        if args.store_attack:
            adversarial_data.append((x_new, target_clean))

        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-inputs_clean).view(args.batch_size, -1), p=2, dim=1)
        l2norm_diff = l2norm_diff/((x_new.view(args.batch_size, -1)).size(1))
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

        # To inject the Adversarial Image into "base model" to compute the "number of non-zeros or net_ones" and "accuracy"
        output_new = model2_in(x_new)
        net_ones = output_new[1]
        total_net_ones_after += net_ones
        total_net_sizes_after += net_sizes
        final_pred = output_new[0].max(dim=1, keepdim=False)[1]
        correct_after += (final_pred == target_clean).sum().item()
        
        num_processed_images += args.batch_size

        if args.save_samples:
            img_grid_clean = torchvision.utils.make_grid(inputs_clean)
            vutils.save_image(img_grid_clean.detach(), './sample_images/clean_idx_%03d.png' %(count), normalize=True)
            img_grid_dirty = torchvision.utils.make_grid(x_new)
            vutils.save_image(img_grid_dirty.detach(), './sample_images/dirty_idx_%03d.png' %(count), normalize=True)

    # To Create a new dataset using the AdversarialDataset class
    if args.store_attack:
        adversarial_dataset = AdversarialDataset(adversarial_data)
    else:
        adversarial_dataset = 0
        
    # Calculate overal accuracy of all test data after sparsity attack
    initial_acc = correct_before/float(num_processed_images)
    final_acc = correct_after/float(num_processed_images)

    print('\nEnded at:', datetime.datetime.now(), "\n")

    return adversarial_dataset, initial_acc, final_acc, l2_norms, (1-(total_net_ones_before/total_net_sizes_before)), (1-(total_net_ones_after/total_net_sizes_after))


if __name__ == '__main__':
       
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="LeNet5 Network with MNIST Dataset")
    parser.add_argument('--batch_size', default=20, type=int, help="Batch size")
    parser.add_argument('--weights', default='../2_copy_weight/lenet5_mnist_fc_one_out.pkl', help="The path to the pre-trained weights")
    parser.add_argument('--dataset', default='../dataset/test', help="The path to the MNIST train and test datasets")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--constrained', action='store_true', help="To apply constrain on the generated purturbed data")
    parser.add_argument('--cons_adv_geneartion', action='store_true', help="Reads advesarial images from previous imax-loop and continuesly optmize the images")
    parser.add_argument('--save_samples', action='store_true', help="To save sample adversarial images")
    parser.add_argument('--store_attack', action='store_true', help="To enable or disable algorithm to store generated adversarials in an output dataset")
    parser.add_argument('--img_index_start', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--img_index_end', default=10000, type=int, help="The last index of the dataset")
    parser.add_argument('--lr', default=0.1, type=float, help="Initial learning rate")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--manualSeed', type=int,  default=13, help='manual seed')
    args = parser.parse_args()
    print('\n', args, '\n')

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("\nRandom Seed: ", args.manualSeed, '\n')
    random.seed(args.manualSeed) # Ensures that the same random numbers are generated every time you run the code with the same seed
    torch.manual_seed(args.manualSeed) # Sets the seed for PyTorch's random number generator
          
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nA {device} assigned for processing!\n")
    
    transform = transforms.ToTensor()
    test_dataset_clean  = torchvision.datasets.MNIST(root=args.dataset, train=False, download=True, transform=transform)
    test_dataset_sub_clean = torch.utils.data.Subset(test_dataset_clean, list(range(args.img_index_start, args.img_index_end)))    
    test_loader_clean = DataLoader(test_dataset_sub_clean, shuffle=False, num_workers=1, batch_size=args.batch_size)
        
    if args.cons_adv_geneartion:
        test_loader_dirty = torch.load('./adversarial_dataset_c100_in.pt', map_location=device)
    else:
        test_dataset_dirty  = torchvision.datasets.MNIST(root=args.dataset, train=False, download=True, transform=transform)
        test_dataset_sub_dirty = torch.utils.data.Subset(test_dataset_dirty, list(range(args.img_index_start, args.img_index_end)))
        test_loader_dirty = DataLoader(test_dataset_sub_dirty, shuffle=False, num_workers=1, batch_size=args.batch_size)
        
    # Model Initialization
    model1_active = lenet5_active(beta=args.beta).to(device)
    model2 = lenet5().to(device)

    if args.save_samples:
        if not os.path.isdir('sample_images'):
            os.mkdir('sample_images')

    if args.weights is not None:
        model1_active.load_state_dict(torch.load(args.weights, map_location=device))
        model2.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()

    num_classes = 10
    c_init = 0.5
    adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before, sr_net_after = Sparsity_Attack_Generation (model1_active, model2, device, test_loader_clean, test_loader_dirty, num_classes, c_init, args)
        
    # To save the generated adversarial dataset to disk
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
    image_out_name = 'adversarial_dataset_mnist'
    image_out_name = os.path.join(image_out_name, '_lr_', str(args.lr), '_', TIME_NOW, '.pt')
    image_out_name = image_out_name.replace('/',  '')
    if args.store_attack:
        torch.save(adversarial_dataset, image_out_name)

    print(f"Test accuracy excluding energy attack: {initial_accuracy}")
    print(f"Test accuracy including energy attack: {final_accuracy}")
    print(f"Sparsity rate before energy attack: {sr_net_before}")
    print(f"Sparsity rate after energy attack: {sr_net_after}")
    print(f"Sparsity reduction applying energy attack: {sr_net_before/(sys.float_info.epsilon if sr_net_after == 0 else sr_net_after)}")
    print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")
    sys.exit(0)

# Arguments (Constrained):   python3 mnist_attack.py --lr 0.1 --eps 0.9 --beta 30 --imax 500 --constrained --store_attack
# Arguments (Unconstrained): python3 mnist_attack.py --lr 0.9 --eps 0.9 --beta 15 --imax 200 --store_attack
