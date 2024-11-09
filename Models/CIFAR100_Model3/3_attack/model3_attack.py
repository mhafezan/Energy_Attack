import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import sys
import argparse
import datetime
import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader, Dataset
from models.source_model3 import Model3_stat
from models.model3 import Model3

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of Cifar100 (min, max)
def clip_tensor(input_tensor, eps_in, batch_size, min_in, max_in):

    with torch.no_grad():
        clapped_in = input_tensor

        for i in range(batch_size):
            torch.clamp(clapped_in[i], min=(min_in[i]-eps_in), max=(max_in[i]+eps_in), out=clapped_in[i])
        
        torch.clamp(clapped_in[:,0, :,:], min=((0-0.4914)/0.2470), max=((1-0.4914)/0.2470), out=clapped_in[:,0, :,:])
        torch.clamp(clapped_in[:,1, :,:], min=((0-0.4822)/0.2435), max=((1-0.4822)/0.2435), out=clapped_in[:,1, :,:])
        torch.clamp(clapped_in[:,2, :,:], min=((0-0.4465)/0.2616), max=((1-0.4465)/0.2616), out=clapped_in[:,2, :,:])
    
    return clapped_in

# Convert a clean image into an advesarial image
def convert_clean_to_adversarial(model_active_in, inputs_dirty, inputs_clean, initial_pred, device, c_init, args):

    c_min = 0
    c_max = 1
    coeff = torch.full((args.batch_size,), c_init, device=device)

    max_clean, _ = torch.max(inputs_clean.view(args.batch_size, -1), dim=1)
    min_clean, _ = torch.min(inputs_clean.view(args.batch_size, -1), dim=1)
    
    """image_loss = nn.MSELoss()"""
    sparsity_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()
    
    x = inputs_dirty.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([x], lr=args.lr, amsgrad=True)
        
    for epoch in range(args.imax):

        # Initialize the x gradients with zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_active_in(x)

        final_pred = outputs[0].max(1, keepdim=False)[1]
        
        coeff = torch.stack([torch.where(initial_pred[i] == final_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
           
        loss1 = torch.stack([classification_loss(outputs[0][j].unsqueeze(0), initial_pred[j].unsqueeze(0)) for j in range(args.batch_size)])

        perfectt = torch.ones(args.batch_size, 1, requires_grad=False).to(device)

        for j in range(args.batch_size):
            assert(outputs[1][j].unsqueeze(0) < 27648)
                
        loss2 = torch.stack([sparsity_loss(outputs[1][j].unsqueeze(0)/27648, perfectt[j].unsqueeze(0)) for j in range(args.batch_size)])

        """loss3 = torch.stack([image_loss(x[j], inputs_clean[j]) for j in range(args.batch_size)])"""

        """loss = coeff * loss1 + loss2 + loss3"""
        loss = coeff * loss1 + loss2

        loss.backward(torch.ones_like(loss))

        optimizer.step()
        
        if args.constrained:
            x = clip_tensor(x, args.eps, args.batch_size, min_clean, max_clean)
 
    return x

def Sparsity_Attack_Generation(model_active_in, model_passive_in, device, test_loader_clean_in, test_loader_dirty_in, num_classes, c_init, args):

    print('\nStarted at:', datetime.datetime.now(), '\n')
    
    model_active_in.eval()
    model_passive_in.eval()

    num_processed_images = 0
    correct_before = 0
    correct_after = 0
    total_net_ones_before = 0
    total_net_sizes_before = 0
    total_net_ones_after = 0
    total_net_sizes_after = 0

    l2_norms = []
    adversarial_data = []

    # net_sizes = 27648 = (3*32*32) + (32*16*16) + (32*16*16) + (8192)
    net_sizes = 27648 * args.batch_size

    # Batch Counter
    count = 0

    for ((data_clean, target_clean) , (data_dirty, target_dirty)) in zip(test_loader_clean_in, test_loader_dirty_in):
        
        inputs_clean, target_clean = data_clean.to(device), target_clean.to(device)
        inputs_dirty, target_dirty = data_dirty.to(device), target_dirty.to(device)

        output_init = model_passive_in(inputs_clean)
        net_ones = output_init[1]
        total_net_ones_before += net_ones
        total_net_sizes_before += net_sizes
        initial_pred = output_init[0].max(1, keepdim=False)[1]
        correct_before += (initial_pred == target_clean).sum().item()
   
        count = count + 1
        if count % 100 == 0:
            print("\ncount=", count)
            print(datetime.datetime.now(), '\n')
        
        x_new = convert_clean_to_adversarial(model_active_in, inputs_dirty, inputs_clean, initial_pred, device, c_init, args)

        if args.store_attack:
            adversarial_data.append((x_new, target_clean))
            
        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (inputs_clean)
        l2norm_diff = torch.norm((x_new-inputs_clean).view(args.batch_size, -1), p=2, dim=1)
        l2norm_diff = l2norm_diff/((x_new.view(args.batch_size, -1)).size(1))
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

        # Re-compute the sparsity rate using the perturbed input
        output_new = model_passive_in(x_new)
        net_ones = output_new[1]
        total_net_ones_after += net_ones
        total_net_sizes_after += net_sizes
        final_pred = output_new[0].max(1, keepdim=False)[1]
        correct_after += (final_pred == target_clean).sum().item()
        
        num_processed_images += args.batch_size

        if args.save_samples:
            img_grid_clean = torchvision.utils.make_grid(inputs_clean)
            vutils.save_image(img_grid_clean.detach(), './sample_images/clean_idx_%03d.png' % (count), normalize=True)
      
            img_grid_dirty = torchvision.utils.make_grid(x_new)
            vutils.save_image(img_grid_dirty.detach(), './sample_images/dirty_idx_%03d.png' % (count), normalize=True)
        
    # To Create a new dataset using the AdversarialDataset class
    if args.store_attack:
        adversarial_dataset = AdversarialDataset(adversarial_data)
    else:
        adversarial_dataset = 0
        

    # Calculate overal accuracy of all test data after sparsity attack
    first_acc = correct_before/float(num_processed_images)
    final_acc = correct_after/float(num_processed_images)

    print('\nEnded at:', datetime.datetime.now(),'\n')

    return adversarial_dataset, first_acc, final_acc, l2_norms, (1-(total_net_ones_before/total_net_sizes_before)), (1-(total_net_ones_after/total_net_sizes_after))



if __name__ == '__main__':
       
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="Model3 Substitute Network with CIFAR100 Dataset")
    parser.add_argument('--batch_size', default=20, type=int, help="Batch size")
    parser.add_argument('--weights', default='../2_copy_weight/model3_cifar100_fc_one_out.pkl', help="The path to the saved weights")
    parser.add_argument('--dataset', default='../cifar100_dataset', help="The path to the cifar100 dataset")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.99, type=float, help="L2-norm bound of epsilon for constraining purturbed image")
    parser.add_argument('--constrained', action='store_true', help="To put constrain on adversarial generation")
    parser.add_argument('--cont_adv_geneartion', action='store_true', help="Reads advesarial images from previous imax-loop and continues optmizing the images")
    parser.add_argument('--save_samples', action='store_true', help="To save generated adversarial samples")
    parser.add_argument('--store_attack', action='store_true', help="To store generated adversarials as a dataset")
    parser.add_argument('--img_index_first', default=0, type=int, help="The first index of the dataset")
    parser.add_argument('--img_index_last', default=10000, type=int, help="The last index of the dataset")
    parser.add_argument('--lr', default=0.01, type=float, help="Initial learning rate")
    parser.add_argument('--imax', default=50, type=int, help="Maximum iterations of the inner loop")
    parser.add_argument('--manualSeed', type=int,  default=13, help='manual seed')
    args = parser.parse_args()
    print(args)

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # CIFAR100 dataset and dataloader declaration
    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    # Dataset definition
    test_dataset_clean = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=TRANSFORM)
    test_dataset_sub_clean = torch.utils.data.Subset(test_dataset_clean,  list(range(args.img_index_first, args.img_index_last)))
    test_loader_clean = DataLoader(test_dataset_sub_clean, batch_size=args.batch_size, shuffle=False)
        
    if args.cont_adv_geneartion:
        test_loader_dirty = torch.load('./adversarial_dataset_cifar100_model3_in.pt', map_location=torch.device('cuda'))
    else:
        test_dataset_dirty = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=TRANSFORM)
        test_dataset_sub_dirty = torch.utils.data.Subset(test_dataset_dirty,  list(range(args.img_index_first, args.img_index_last)))
        test_loader_dirty = DataLoader(test_dataset_sub_dirty, batch_size=args.batch_size, shuffle=False)

    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing!\n")
    
    # Network Initialization
    model_active = Model3(args=args)
    model_passive = Model3_stat()
    model_active.to(device)
    model_passive.to(device)

    if args.save_samples:
        if not os.path.isdir('sample_images'):
            os.mkdir('sample_images')

    if args.weights is not None:
        model_active.load_state_dict(torch.load(args.weights, map_location=device))
        model_passive.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()
    
    num_classes  = len(test_dataset_clean.classes)
    c_init = 0.5
    
    adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before, sr_net_after = Sparsity_Attack_Generation (model_active, model_passive, device, test_loader_clean, test_loader_dirty, num_classes, c_init, args)

    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
    image_out_name = 'adversarial_dataset_cifar100_model3'
    image_out_name = os.path.join(image_out_name, '_lr_', str(args.lr), '_', TIME_NOW, '.pt')
    image_out_name = image_out_name.replace('/',  '')
    
    # Save the generated adversarial dataset to disk
    if args.store_attack:
        torch.save(adversarial_dataset, image_out_name)

    print(f"Test accuracy excluding energy attack: {initial_accuracy}")
    print(f"Test accuracy including energy attack: {final_accuracy}")
    print(f"Sparsity rate before energy attack: {sr_net_before}")
    print(f"Sparsity rate after energy attack: {sr_net_after}")
    print(f"Sparsity reduction applying energy attack: {sr_net_before/(sys.float_info.epsilon if sr_net_after == 0 else sr_net_after)}")
    print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")
    
    f_ptr = open('./tmp_sparsity_results.txt','a')
    print('lrrrrrrr=', args.lr, file=f_ptr)
    print('accu_before=', initial_accuracy, file=f_ptr)
    print('accu_after=', final_accuracy, file=f_ptr)
    print('sparsity_before=', sr_net_before, file=f_ptr)
    print('sparsity_after=', sr_net_after, file=f_ptr)
    print(' ', file=f_ptr)
    
    sys.exit(0)

# Arguments:
# python3 model3_attack.py --lr=0.001 --eps 0.99 --beta 50 --imax 50 --constrained --store_attack
# python3 model3_attack.py --lr=0.08  --eps 0.99 --beta 5  --imax=50 --store_attack
