# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py
# https://github.com/mnswdhw/DefenseGAN-and-Cowboy-Defense/blob/main/defense.py

from __future__ import print_function
import argparse
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import datetime
from torch.utils.data import DataLoader, Dataset
from model import Discriminator, Generator, AlexNet

class Z_Stars_Dataset (Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class EmptyZ (Dataset):
    def __init__(self, dataset_size, nz, transform=None):
        self.dataset_size = dataset_size   # Number of samples = 10000 with the same size of CIFAR-10 testset
        self.nz = nz                       # Size of the Z vector
        self.transform = transform         # Any transformations to apply

        # Define the shape of the empty image tensor
        self.image_shape = (self.nz, 1, 1)  
        # Empty image tensor (filled with zeros)
        self.empty_image = torch.zeros(self.image_shape)
        # CIFAR-10 has 10 classes, so use zero as the dummy label
        self.empty_label = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Apply the transform to the empty image, if any
        image = self.empty_image
        if self.transform:
            image = self.transform(image)

        # Return the empty image and a dummy label (e.g., 0)
        return image, self.empty_label

def optimize_for_L_iterations(modelG, data, z_hat, loss, args):
    
    z_hat = z_hat.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([z_hat], lr=args.lr, amsgrad=True)

    # To backup copies of reconstruct_loss and z_hat for each loop iteration below
    backup_reconstruct_loss = torch.Tensor(args.rec_iter, data.size(0))
    backup_z_hat = torch.Tensor(args.rec_iter, z_hat.size(0), z_hat.size(1), z_hat.size(2), z_hat.size(3))
    min_z_hat = torch.empty_like(z_hat)
    
    # L steps of optimization to find the optimum Z for Generator w.r.t ||G(z)-X||
    for iteration in range(args.rec_iter):
            
        optimizer.zero_grad()
            
        fake_image = modelG(z_hat)            
          
        if args.use_loss_per_image:
            reconstruct_loss = torch.mean((fake_image - data)**2, dim=tuple(range(1, fake_image.ndim))) # To compute MSE for each image independently
            reconstruct_loss.backward(torch.ones_like(reconstruct_loss))
        else:
            reconstruct_loss = loss(fake_image, data)
            reconstruct_loss.backward()
            
        optimizer.step()

        # To store backup copies (applicable for both per image and per batch)
        backup_reconstruct_loss[iteration] = reconstruct_loss
        backup_z_hat[iteration] = z_hat

    min_reconstruct_loss = torch.min(backup_reconstruct_loss, dim=0)

    for b in range(data.size(0)):
        min_z_hat[b] = backup_z_hat[min_reconstruct_loss.indices[b]][b]

    """min_z_hat = min_z_hat.clone().detach().requires_grad_(True)"""
    
    # To free memory by deleting the backup tensors
    del backup_reconstruct_loss, backup_z_hat

    return min_z_hat

def get_z_sets (modelG, modelD, data, z_items, lr, loss, device, rec_iter = 500, rec_rr = 10, input_latent = 100):

    modelG.eval()
    modelD.eval()
    
    z_hats_recs = torch.Tensor(rec_rr, data.size(0), input_latent, 1, 1)

    for idx in range(rec_rr):

        if(args.cont_z_geneartion):
            z_hat = z_items[idx]
        else:
            z_hat = torch.randn(data.size(0), input_latent, 1, 1).to(device)

        z_hat = z_hat.clone().detach().requires_grad_(False)
        
        z_hat = optimize_for_L_iterations(modelG, data, z_hat, loss, args)
            
        z_hats_recs[idx] = z_hat.cpu().detach().clone()

    z_hats_recs = z_hats_recs.clone().detach().requires_grad_(False)

    return z_hats_recs

def select_z_star(model, data, z_hats_recs, loss, device):
    
    z_losses = torch.Tensor(args.rec_rr)
    
    for i in range(args.rec_rr):
        
        z = model(z_hats_recs[i].to(device))
        
        """z = z.view(-1, data.size(1), data.size(2), data.size(3))"""
        
        z_losses[i] = loss(z, data).cpu().item()
        
    min_idx = torch.argmin(z_losses)
    
    return z_hats_recs[min_idx]

def defense_on_clean_image (model, ModelG, ModelD, test_loader, z_sets_loader, device, args):
 
    model.eval()
    corrects_dirty = 0
    corrects_clean = 0
    num_processed_images = 0
    
    loss = nn.MSELoss()
    
    display_steps = 1
    
    net_sizes = 0
    net_ones_original = 0
    net_ones_reconstruct = 0
    
    z_sets_array = []
    z_stars_array = []
    
    # Transformation is now done here for data and data-hat
    my_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    batch_idx = 0
    
    for ((inputs, labels) , (z_items, z_labels)) in zip(test_loader, z_sets_loader):
    
        batch_idx += 1
        
        data = inputs.to(device)
    
        # To find the optimal z* for rec_rr number of random z vectors
        z_sets = get_z_sets (ModelG, ModelD, data, z_items, args.lr, loss, device, rec_iter = args.rec_iter, rec_rr = args.rec_rr, input_latent = args.nz)
    
        # To store a z_set for each Batch; Same batch size should be considered when loading z_sets_array
        z_sets_array.append((z_sets, labels))
    
        # To select the best z*
        z_star = select_z_star (ModelG, data, z_sets, loss, device)
    
        # To store the optimal z* per image and creating a dataset with the same length of CIFAR10
        for b in range(z_star.size(0)):
            temp = z_star[b]
            z_stars_array.append((temp, labels[b]))
    
        # To pass the z* to Generator
        data_hat = ModelG(z_star.to(device)).cpu().detach()
    
        # To evaluate data_hat=G(z*) with the LeNet5 model in terms of accuracy and sparsity rate
        data_hat = data_hat.to(device)
        labels = labels.to(device)
        
        # For adversarial data
        data_hat = my_transform(data_hat)
        outputs_dirty = model(data_hat)
        net_ones_reconstruct += outputs_dirty[1]
        net_sizes += (458624*args.batch_size)
        _, preds_dirty = torch.max(outputs_dirty[0], 1)
        corrects_dirty += torch.sum(preds_dirty == labels.data)
        num_processed_images += args.batch_size
    
        # For clean data
        data_rescaled = my_transform(data)
        outputs_clean = model(data_rescaled)
        net_ones_original += outputs_clean[1]
        _, preds_clean = torch.max(outputs_clean[0], 1)
        corrects_clean += torch.sum(preds_clean == labels.data)
    
        if batch_idx % display_steps == 0:
            print('# Batch: ', batch_idx, '/', len(test_loader), ', Accuracy (dirty): ', corrects_dirty.item()/num_processed_images, ', SR (dirty): ', (1-(net_ones_reconstruct/net_sizes)))
            print('# Batch: ', batch_idx, '/', len(test_loader), ', Accuracy (clean): ', corrects_clean.item()/num_processed_images, ', SR (clean): ', (1-(net_ones_original/net_sizes)), '\n')
            
        del data, data_hat, z_star, labels, outputs_dirty, preds_dirty, outputs_clean, preds_clean
    
    test_acc_dirty = corrects_dirty.item() / num_processed_images
    test_acc_clean = corrects_clean.item() / num_processed_images
    
    print('\n************************* Final Statistics ************************\n')
    print('rec_iter: ', args.rec_iter, ', rec_rr = ', args.rec_rr,'\n')
    print('Accuracy (clean): ', test_acc_clean, ', Accuracy (dirty): ', test_acc_dirty,'\n')
    print ('Sparsity (clean): ', (1-(net_ones_original/net_sizes)), ', Sparsity (dirty): ', (1-(net_ones_reconstruct/net_sizes)), '\n')
    print('\n*******************************************************************\n')
    
    ######################### Saving z_sets #########################
    
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
    z_sets_name = 'z_sets_dataset'
    z_sets_name = os.path.join(z_sets_name, '_lr_', str(args.lr), '_', TIME_NOW, '.pt')
    z_sets_name = z_sets_name.replace('/',  '')
    z_sets_name = './z_sets/'+ z_sets_name
    
    z_sets_array_dataset = Z_Stars_Dataset(z_sets_array)
    
    if not os.path.isdir("z_sets"):
        os.mkdir("z_sets")
      
    torch.save(z_sets_array_dataset, z_sets_name)
    
    ######################### Saving z_star #########################
    
    z_stars_array_dataset = Z_Stars_Dataset(z_stars_array)
    
    if not os.path.isdir("z_stars"):
        os.mkdir("z_stars")
      
    torch.save(z_stars_array_dataset, './z_stars/zstar_dataset.pt')
    
    return



if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='../cifar10_dataset', help='Path to the CIFAR10 dataset')
    parser.add_argument('--batch_size', type=int, default=20, help='Input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='The height or width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='The size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--rec_iter', type=int, default=100, help='The number of iterations to optimize z')
    parser.add_argument('--rec_rr', type=int, default=20, help='The number of Random Restarts (RR) or random initializations of z')
    parser.add_argument('--lr', type=float, default=0.2, help='The learning rate to find optimal Z*')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--manualSeed', type=int,  default=13, help='Manual seed')
    parser.add_argument('--gen_path', default='../1_train/weight_dir/netG_epoch_99.pth', help='Path to model_generator trained model')
    parser.add_argument('--disc_path', default='../1_train/weight_dir/netD_epoch_99.pth', help='Path to model_discriminator trained model')
    parser.add_argument('--cifar10_weights', default='../../../../Ehsan/CIFAR10_AlexNet/3_copy_weight/alexnet_cifar10_fc_one_out_458624.pkl', help='Path to AlexNet pre-trained model on Cifar10')
    parser.add_argument('--cont_z_geneartion', action='store_true', help="Use z_sets from previous round to start with")
    parser.add_argument('--image_index_first', type=int, default=0, help='The first index of the dataset')
    parser.add_argument('--image_index_last', type=int, default=10000, help='The last index of the dataset')
    parser.add_argument('--use_loss_per_image', action='store_true', help="The loss function is computed per image not per batch")    
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
    
    gen_model = Generator(ngpu, nz=args.nz, ngf=args.ngf).to(device)   
    gen_model.load_state_dict(torch.load(args.gen_path, map_location=device))

    disc_model = Discriminator(ngpu, ndf=args.ndf).to(device)
    disc_model.load_state_dict(torch.load(args.disc_path, map_location=device))

    # To define and load AlexNet model from Adversarial Attack Algorithm
    model_cifar10 = AlexNet().to(device)
    model_cifar10.load_state_dict(torch.load(args.cifar10_weights, map_location=device))

    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    """
    # Transformation is not used for training and testing of RAW CIFAR10
    dataset = dset.CIFAR10(root=args.dataset, download=True, train=False, transform=transforms.ToTensor())
    dataset_sub = torch.utils.data.Subset(dataset, list(range(args.image_index_first, args.image_index_last)))
    dataset_loader = DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False)
    print(f"\nDataloader size is {len(dataset_loader)} considering batch size of {args.batch_size}\n")

    if args.cont_z_geneartion:
        z_sets_loader = torch.load('./z_sets/zsets_dataset_in.pt', map_location=torch.device('cuda'))
    else:
        # Create an empty CIFAR-10-like dataset with the same size as the testset
        z_dataset = EmptyZ(dataset_size=len(dataset), nz=args.nz)
        z_dataset_sub = torch.utils.data.Subset(z_dataset, list(range(args.image_index_first, args.image_index_last)))
        z_sets_loader = DataLoader(z_dataset_sub, batch_size=args.batch_size, shuffle=False)
  
    defense_on_clean_image (model_cifar10, gen_model, disc_model, dataset_loader, z_sets_loader, device, args)

    print('\nEnded at:', datetime.datetime.now(),'\n')
    sys.exit(0)

"""    
python3 test_clean.py --gen_path ../1_train/weight_dir/netG_epoch_99.pth --disc_path ../1_train/weight_dir/netD_epoch_99.pth --use_loss_per_image --nz 512 --ngf 128 --ndf 128 --lr 0.2 --rec_iter 100 --rec_rr 20

# To continue from the last round:
python3 test_clean.py --gen_path ../1_train/weight_dir/netG_epoch_99.pth --disc_path ../1_train/weight_dir/netD_epoch_99.pth --cont_z_geneartion --nz 512 --ngf 128 --ndf 128 --lr 0.2 --rec_iter 100 --rec_rr 20
"""