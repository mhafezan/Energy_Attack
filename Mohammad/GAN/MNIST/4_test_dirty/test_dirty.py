# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

#https://github.com/csinva/gan-vae-pretrained-pytorch/blob/master/mnist_dcgan
#https://github.com/pytorch/examples/tree/main/dcgan
#https://github.com/mnswdhw/DefenseGAN-and-Cowboy-Defense/tree/main

# python dcgan.py --dataset mnist --dataroot /scratch/users/vision/yu_dl/raaz.rsk/data/cifar10 --imageSize 28 --cuda --outf . --manualSeed 13 --niter 100

from __future__ import print_function
import argparse
import os, sys
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import datetime
from torch.utils.data import DataLoader, Dataset
from model import Discriminator, Generator, LeNet5

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return  self.data[idx]

def optimize_for_L_iterations (z_hat, rec_iter, modelG, data, loss, lr):
    
    z_hat = z_hat.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([z_hat], lr=lr, amsgrad=True)

    # L steps of optimization to find the optimum Z for Generator w.r.t X and G(Z)
    for iteration in range(rec_iter):
            
        optimizer.zero_grad()
            
        fake_image = modelG(z_hat)            
        fake_image = fake_image.view(-1, data.size(1), data.size(2), data.size(3))                  
          
        reconstruct_loss = loss(fake_image, data)
        reconstruct_loss.backward()
            
        optimizer.step()

    return z_hat

def get_z_sets_defensegan (modelG, modelD, data, lr, loss, device, rec_iter = 500, rec_rr = 10, input_latent = 100):

    modelG.eval()
    modelD.eval()
    
    z_hats_recs = torch.Tensor(rec_rr, data.size(0), input_latent, 1, 1)
    
    z_hats_orig = torch.Tensor(rec_rr, data.size(0), input_latent, 1, 1)

    for idx in range(len(z_hats_recs)):
        
        z_hat = torch.randn(data.size(0), input_latent, 1, 1).to(device)
        
        z_hat = z_hat.clone().detach().requires_grad_(False)

        z_hats_orig[idx] = z_hat.cpu().detach().clone()
        
        z_hat = optimize_for_L_iterations(z_hat, rec_iter, modelG, data, loss, lr)
            
        z_hats_recs[idx] = z_hat.cpu().detach().clone()
        

    z_hats_orig = z_hats_orig.clone().detach().requires_grad_(False)
    z_hats_recs = z_hats_recs.clone().detach().requires_grad_(False)

    return z_hats_orig, z_hats_recs

def select_z_star (model, data, z_hats, loss, device):
    
    z_losses = torch.Tensor(len(z_hats))
    
    for i in range(len(z_hats)):
        
        z = model(z_hats[i].to(device))
        
        z = z.view(-1, data.size(1), data.size(2), data.size(3))
        
        z_losses[i] = loss(z, data).cpu().item()
        
    min_idx = torch.argmin(z_losses)
    
    return z_hats[min_idx]

def defense_on_clean_image (model, ModelG, ModelD, test_loader, device, args):
 
    model.eval()
    corrects_original = 0
    corrects_reconstruct = 0
    num_processed_images = 0
    
    loss = nn.MSELoss()
    
    display_steps = 20

    net_sizes = 0 # 2108*args.batch_size
    net_ones_original = 0
    net_ones_reconstruct = 0
  
    for batch_idx, (inputs, labels, adversarial) in enumerate(test_loader):

        inputs = inputs.clone().detach().requires_grad_(False)
      
        data = inputs.to(device)
        labels = labels.to(device)

        # To find the optimal z* set for rec_rr number of random z vectors
        _, z_sets = get_z_sets_defensegan (ModelG, ModelD, data, args.lr, loss, device, rec_iter = args.rec_iter, rec_rr = args.rec_rr, input_latent = args.z_dim)

        # To select the best z*
        z_star = select_z_star (ModelG, data, z_sets, loss, device)

        # To obtain data_hat = G(z*)
        data_hat = ModelG(z_star.to(device)).cpu().detach()
        data_hat = data_hat.to(device)
        
        recon_imgs = data_hat[:5]
        img_grid = torchvision.utils.make_grid(recon_imgs)
        vutils.save_image(img_grid.detach(), './fake_sample_images/fake_samples_idx_%03d.png' % (batch_idx), normalize=True)
        img_grid = torchvision.utils.make_grid(inputs[:5])
        vutils.save_image(img_grid.detach(), './real_sample_images/real_samples_idx_%03d.png' % (batch_idx), normalize=True)

        # To evaluate sparsity and accuracy of RECONSTRUCTED DATA using LeNet5 model
        outputs_reconstruct = model(data_hat)
        net_ones_reconstruct += outputs_reconstruct[1]
        net_sizes += (2108*args.batch_size)
        _, preds_reconstruct = torch.max(outputs_reconstruct[0], 1)
        corrects_reconstruct += torch.sum(preds_reconstruct == labels.data)
        num_processed_images += args.batch_size

        if batch_idx % display_steps == 0:
            print('# Batch: ', batch_idx+1, '/', len(test_loader), ', Accuracy (reconstructed): ', corrects_reconstruct.item()/num_processed_images, ', Sparsity (reconstructed): ', (1-(net_ones_reconstruct/net_sizes)),'\n')

        # To evaluate sparsity and accuracy of REAL DATA using LeNet5 model
        outputs_original = model(data)
        net_ones_original += outputs_original[1]
        _, preds_original = torch.max(outputs_original[0], 1)
        corrects_original += torch.sum(preds_original == labels.data)

        if batch_idx % display_steps == 0:
            print('# Batch: ', batch_idx+1, '/', len(test_loader), ', Accuracy (clean): ', corrects_original.item() / num_processed_images, ', Sparsity (clean): ', (1-(net_ones_original/net_sizes)),'\n')
 
        del data, data_hat, labels, outputs_reconstruct, preds_reconstruct, outputs_original, preds_original, z_star

    test_acc_dirty = corrects_reconstruct.item() / num_processed_images
    test_acc_clean = corrects_original.item() / num_processed_images

    print('\n********************** Final Stats ****************************\n')
    print('rec_iter = ', args.rec_iter, 'rec_rr = ', args.rec_rr, 'accu_clean = ', test_acc_clean, 'accu_dirty = ', test_acc_dirty)
    print('sparsity_clean = ', (1-(net_ones_original/net_sizes)), 'sparsity_dirty = ', (1-(net_ones_reconstruct/net_sizes)),'\n')

    return


    
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--imageSize', type=int, default=28, help='The height or width of the input image to network')
    parser.add_argument('--z_dim', type=int, default=100, help='The size of the latent z vector')
    parser.add_argument('--rec_iter', type=int, default=500, help='The number of iterations to optimize z')
    parser.add_argument('--rec_rr', type=int, default=10, help='The number of Random Restarts (RR) or random initializations of z')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.09, help='The learning rate to find optimal Z*')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--manualSeed', type=int,  default=13, help='Manual Seed')
    parser.add_argument('--model_gen_path', default='../1_train/weight_dir/netG_epoch_97.pth', help='Path to model_generator trained model')
    parser.add_argument('--model_disc_path', default='../1_train/weight_dir/netD_epoch_97.pth', help='Path to model_discriminator trained model')
    parser.add_argument('--mnist_weights', default='../../../../Ehsan/MNIST_LeNet5/2_copy_weight/lenet5_mnist_fc_one_out.pkl', help='Path to MNIST pre-trained model')
    parser.add_argument('--im_index_fisrt', type=int, default=0, help='The fisrt index of the dataset')
    parser.add_argument('--im_index_last', type=int, default=10000, help='The last index of the dataset')    
    args = parser.parse_args()
    print(f"\n{args}\n")

    print('Started at:', datetime.datetime.now(),'\n')

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed:", args.manualSeed,'\n')
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # Device Initialization
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    ngpu = args.ngpu

    model_gen = Generator(ngpu).to(device)   
    model_gen.load_state_dict(torch.load(args.model_gen_path, map_location=device))

    model_disc = Discriminator(ngpu).to(device)   
    model_disc.load_state_dict(torch.load(args.model_disc_path, map_location=device))

    # To define and load MNIST model from Adversarial Attack Algorithm
    model_mnist = LeNet5().to(device)
    model_mnist.load_state_dict(torch.load(args.mnist_weights, map_location=device))
    model_mnist.eval()

    dataset = torch.load('../3_convert_dirt_dataset_batch_one/adversarial_dataset_mnist_unconstrained_lr_0.9_bs_1.pt', map_location=torch.device('cpu'))
    dataset_sub = torch.utils.data.Subset(dataset, list(range(args.im_index_fisrt, args.im_index_last)))
    dataset_loader = DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False)
    print(f"\nData loader size is {len(dataset_loader)} considering batch size of {args.batch_size}\n")
  
    if not os.path.isdir('real_sample_images'):
        os.mkdir('real_sample_images')

    if not os.path.isdir('fake_sample_images'):
        os.mkdir('fake_sample_images')

    defense_on_clean_image (model_mnist, model_gen, model_disc, dataset_loader, device, args)

    print('\nEnded at:', datetime.datetime.now(),'\n')
    sys.exit(0)

# Arguments: python3 test_dirty.py --rec_iter 900 --rec_rr 10 --z_dim 100