import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import os
import sys
import argparse
import torch.nn.functional as F
import datetime
import torchvision.utils as vutils
import random
from torch.utils.data import DataLoader, Dataset
from models.source_model2 import Model2_stat
from models.model2 import Model2

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of Cifar10 (min, max)
def clip_tensor(input_tensor, eps_in, batch_size, input_tensor_clean, min_in, max_in):

    with torch.no_grad():
        clapped_in = input_tensor

        for i in range(batch_size):
            torch.clamp(clapped_in[i], min=(min_in[i]-eps_in), max=(max_in[i]+eps_in), out=clapped_in[i])
        
        torch.clamp(clapped_in[:,0, :,:], min=((0-0.485)/0.229), max=((1-0.485)/0.229) , out=clapped_in[:,0, :,:])
        torch.clamp(clapped_in[:,1, :,:], min=((0-0.456)/0.224), max=((1-0.456)/0.224), out=clapped_in[:,1, :,:])
        torch.clamp(clapped_in[:,2, :,:], min=((0-0.406)/0.225), max=((1-0.406)/0.225), out=clapped_in[:,2, :,:])
    
    return clapped_in

# Convert a clean image into an advesarial image
def convert_clean_to_adversarial(model_active_in, device, inputs_dirty, x_clean, init_pred, c_init, args):

    c_min = 0
    c_max = 1
    coeff = torch.full((args.batch_size,), c_init, device=device)

    max_clean, _ = torch.max(x_clean.view(args.batch_size, -1), dim=1)
    min_clean, _ = torch.min(x_clean.view(args.batch_size, -1), dim=1)

    x = inputs_dirty.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([x], lr=args.lr, amsgrad=True)

    image_loss = nn.MSELoss()
        
    for epoch in range(args.imax):            

        optimizer.zero_grad()
            
        # forward + backward + optimize
        outputs = model_active_in(x)

        final_pred = outputs[0].max(1, keepdim=False)[1]
        
        coeff = torch.stack([torch.where(init_pred[i] == final_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
 
        loss1 = torch.stack([F.cross_entropy(outputs[0][j].unsqueeze(0), init_pred[j].unsqueeze(0)) for j in range(args.batch_size)])

        criteria_sparse_loss = nn.MSELoss()

        perfectt = torch.ones(args.batch_size, 1, requires_grad=False).to(device)

        for j in range(args.batch_size):
            assert(outputs[1][j].unsqueeze(0) < 458624)
                
        loss2 = torch.stack([criteria_sparse_loss(outputs[1][j].unsqueeze(0)/458624, perfectt[j].unsqueeze(0)) for j in range(args.batch_size)])

        loss3 = torch.stack([image_loss(x[j], x_clean[j]) for j in range(args.batch_size)])

        loss = coeff * loss1 + loss2 + loss3

        loss.backward(torch.ones_like(loss))

        optimizer.step()
        
        if args.constrained:
            x = clip_tensor(x, args.eps, args.batch_size, x_clean, min_clean, max_clean)
        
    return x

def Sparsity_Attack_Generation(model_active_in, model2_in, device, test_loader_clean_in, test_loader_dirty_in, num_classes, c_init, args):

    print('\nStarted at:', datetime.datetime.now(), '\n')
    
    model_active_in.eval()
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

    # net_sizes = 458624 = 6*24*24 + 16*8*8 + 120 + 84 + 10
    net_sizes = 458624 * args.batch_size

    count = 0

    for ((data_clean, target_clean) , (data_dirty, target_dirty)) in zip(test_loader_clean, test_loader_dirty):
        inputs_clean, target_clean = data_clean.to(device), target_clean.to(device)
        inputs_dirty, target_dirty = data_dirty.to(device), target_dirty.to(device)

        output_init = model2_in(inputs_clean)
        net_ones = output_init[1]
        total_net_ones_before += net_ones
        total_net_sizes_before += net_sizes
        init_pred = output_init[0].max(1, keepdim=False)[1]
        correct_before += (init_pred == target_clean).sum().item()
   
        count = count +1

        if((count%200)==0):
            print("count=", count)
            print(datetime.datetime.now() )
        
        x_new = convert_clean_to_adversarial (model_active_in, device, inputs_dirty, inputs_clean, init_pred, c_init, args)

        if args.store_attack:
            adversarial_data.append((x_new, target_clean))
            
        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-inputs_clean).view(args.batch_size, -1), p=2, dim=1)
        l2norm_diff = l2norm_diff/((x_new.view(args.batch_size, -1)).size(1))
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

        # Re-compute the sparsity rate using the perturbed input
        output_new = model2_in(x_new)
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
    parser = argparse.ArgumentParser(description="Model2 Substitute Network with CIFAR10 Dataset")
    parser.add_argument('--batch_size', default=20, type=int, help="Batch size")
    parser.add_argument('--weights', default='../2_copy_weight/model2_cifar10_fc_one_out.pkl', help="The path to the saved weights")
    parser.add_argument('--dataset', default='../cifar10_dataset', help="The path to the cifar10 dataset")
    parser.add_argument('--beta', default=50, type=int, help="Beta parameter used in Tanh function")
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

    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])
    
    # Dataset definition
    test_dataset_clean  = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform)
    test_dataset_sub_clean = torch.utils.data.Subset(test_dataset_clean,  list(range(args.img_index_first, args.img_index_last)))        
    test_loader_clean = DataLoader(test_dataset_sub_clean, shuffle=False, num_workers=1, batch_size=args.batch_size) 
        
    if args.cont_adv_geneartion:
        test_loader_dirty = torch.load('./adversarial_dataset_cifar10_in.pt', map_location=torch.device('cuda'))
    else:
        test_dataset_dirty  = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform)
        test_dataset_sub_dirty = torch.utils.data.Subset(test_dataset_dirty,  list(range(args.img_index_first, args.img_index_last)))        
        test_loader_dirty = DataLoader(test_dataset_sub_dirty, shuffle=False, num_workers=1, batch_size=args.batch_size)

    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n A {device} is assigned for processing! \n")
    
    # Network Initialization
    model_active = alexnet_active(beta=args.beta)
    model2 = alexnet()
    model_active.to(device)
    model2.to(device)

    if args.save_samples:
        if not os.path.isdir('sample_images'):
            os.mkdir('sample_images')

    if args.weights is not None:
        model_active.load_state_dict(torch.load(args.weights, map_location=device))
        model2.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print('No weights are provided.')
        sys.exit()
    
    num_classes  = 10
    c_init = 0.5
    
    adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total = Sparsity_Attack_Generation (model_active, model2, device, test_loader_clean, test_loader_dirty,num_classes, c_init, args)

    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
    image_out_name = 'adversarial_dataset_cifar10'
    image_out_name = os.path.join(image_out_name, '_lr_', str(args.lr), '_', TIME_NOW, '.pt')
    image_out_name = image_out_name.replace('/',  '')
    
    # Save the generated adversarial dataset to disk
    if args.store_attack:
        torch.save(adversarial_dataset, image_out_name)

    print(f"Test accuracy excluding energy attack: {initial_accuracy}")
    print(f"Test accuracy including energy attack: {final_accuracy}")
    print(f"Sparsity rate before energy attack: {sr_net_before_total}")
    print(f"Sparsity rate after energy attack: {sr_net_after_total}")
    print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
    print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")

# Arguments:
# python3 model2_attack.py --lr=0.001 --eps 0.99 --constrained --store_attack
# python3 model2_attack.py --lr=0.08 beta=5 imax=50 --store_attack