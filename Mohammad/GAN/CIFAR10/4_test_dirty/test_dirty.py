# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

from __future__ import print_function
import argparse
import os
import random
import torch
import torchvision

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import math
import datetime
from torch.utils.data import DataLoader

from model import Discriminator, Generator, AlexNet
from torch.utils.data import DataLoader, random_split,Dataset

#https://github.com/mnswdhw/DefenseGAN-and-Cowboy-Defense/blob/main/defense.py

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

 
class Z_Stars_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



def  optimize_for_one_iteration(z_hat, rec_iter, modelG, data, loss, lr, args):
    z_hat = z_hat.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([z_hat], lr=lr, amsgrad=True) #, weight_decay=1e-4)
    #cur_lr = lr

    # the main optimization loop 
    for iteration in range(rec_iter):
            
        optimizer.zero_grad()
            
        fake_image = modelG(z_hat)            
        fake_image = fake_image.view(-1, data.size(1), data.size(2), data.size(3))                  
          
  
        if args.use_loss_per_image:
            reconstruct_loss = torch.mean((fake_image - data)**2, dim=tuple(range(1, fake_image.ndim)))
            reconstruct_loss.backward(torch.ones_like(reconstruct_loss))
        else:
            reconstruct_loss = loss(fake_image,data)
            reconstruct_loss.backward()

            
        optimizer.step()

        #cur_lr = adjust_lr(optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter= rec_iter)

    
    z_hat = z_hat.clone().detach().requires_grad_(False)

    return z_hat







def get_z_sets_defensegan(modelG,modelD, data, z_sets_item, lr, loss, device, rec_iter = 500, rec_rr = 10, input_latent = 100, global_step = 1):

    modelG.eval()
    modelD.eval()
    
    data = data.clone().detach().requires_grad_(False)

    z_hats_recs = torch.Tensor(rec_rr, data.size(0), input_latent,1,1)
    
    z_hats_orig = torch.Tensor(rec_rr, data.size(0), input_latent,1,1)


    for idx in range(len(z_hats_recs)):

        if(args.cont_z_geneartion):
            z_hat = z_sets_item[idx]
        else:
            z_hat = torch.randn(data.size(0), input_latent,1,1).to(device)


        z_hat = z_hat.clone().detach().requires_grad_(False)

        z_hats_orig[idx] = z_hat.cpu().detach().clone()
        
        #aaaaaaaaaaa
        z_hat = optimize_for_one_iteration(z_hat, rec_iter, modelG, data, loss, lr, args)
            
        z_hats_recs[idx] = z_hat.cpu().detach().clone()
        

    z_hats_orig = z_hats_orig.clone().detach().requires_grad_(False)
    z_hats_recs = z_hats_recs.clone().detach().requires_grad_(False)

    return z_hats_orig, z_hats_recs


def get_z_star(model, data, z_hats_recs, loss, device):
    
    reconstructions = torch.Tensor(len(z_hats_recs))
    
    for i in range(len(z_hats_recs)):
        
        z = model(z_hats_recs[i].to(device))
        
        z = z.view(-1, data.size(1), data.size(2), data.size(3))
        
        reconstructions[i] = loss(z, data).cpu().item()
        
    min_idx = torch.argmin(reconstructions)
    
    return z_hats_recs[min_idx]


def defense_on_clean_img(model, ModelG, ModelD, test_loader, z_sets_loader, device, args):  #sigma, is_defense_gan = False):
 
  model.eval()
  running_corrects = 0
  running_corrects_clean = 0
  epoch_size = 0

  loss = nn.MSELoss()
  rec_iter = args.rec_iter #400 #900 #bbbbbbbbbbbbbbbbb
  rec_rr = args.rec_rr  #20 #10
  INPUT_LATENT = args.nz #100 
  global_step = 3.0

  display_steps = 1

  #sparsity stat
  net_sizes = 0 #458624*args.batch_size
  
  net_ones_orig = 0
  net_ones_reconstruct = 0

  z_sets_array = []
  z_stars_array = []


  transform_to_256 = transforms.Compose([
      transforms.Resize(256),
      #transforms.CenterCrop(224),
      #transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
           

  my_transform1 = transforms.Compose([
      #transforms.Resize(32),
      #transforms.CenterCrop(224),
      #transforms.ToTensor(),
      transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225])
  ])
            
  my_transform2 = transforms.Compose([
      transforms.Resize(32),
      #transforms.CenterCrop(224),
      #transforms.ToTensor(),
      transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
  ]) 
 

      

  batch_idx = 0
      
  for ((inputs, labels, adversarial) , (z_sets_item, z_label)) in zip(test_loader, z_sets_loader):

      batch_idx += 1

      backup_inputs = inputs
      
      inputs = my_transform1(inputs)
      inputs = my_transform2(inputs)

      inputs = torch.clamp(inputs, 0, 1) #adversarial images in unconstrained mode may go beyond [0, 1]. GAN was trained for pixels between [0, 1].
      
      data = inputs.to(device)

      # find z*
      #if is_defense_gan == True:
      _, z_sets = get_z_sets_defensegan(ModelG,ModelD, data, z_sets_item, args.lr, \
                                        loss, device, rec_iter = rec_iter, \
                                        rec_rr = rec_rr, input_latent = INPUT_LATENT, global_step = global_step)    
      #else:    
      #  _, z_sets = get_z_sets_cowboy(ModelG,ModelD, data, learning_rate, \
      #                              loss, device, rec_iter = rec_iter, \
      #                              rec_rr = rec_rr, input_latent = INPUT_LATENT, global_step = global_step,sigma = sigma)

      z_sets_array.append((z_sets, labels))


      z_star = get_z_star(ModelG, data, z_sets, loss, device)

      for ii in range(z_star.size(0)):
          tmpp = z_star[ii]
          #tmpp = tmpp.unsqueeze(0)

          z_stars_array.append((tmpp, labels[ii]))

      
      # generate data
      data_hat = ModelG(z_star.to(device)).cpu().detach()


      # classifier 
      data_hat = data_hat.to(device)

      labels = labels.to(device)

      # evaluate

      data_hat = transform_to_256(data_hat)
      
      outputs = model(data_hat)
      recon_imgs = data_hat[:5]
      
      #my_data_hat = my_transform(data_hat)
      #outputs = model(my_data_hat)
      #recon_imgs = my_data_hat[:5]

      net_ones_reconstruct += outputs[1]
      net_sizes += (458624*args.batch_size)
      
      img_grid = torchvision.utils.make_grid(recon_imgs)

      vutils.save_image(img_grid.detach(),
                        './fake_sample_images/fake_samples_idx_%03d.png' % (batch_idx),
                        normalize=True)
      
      
 
      
      _, preds = torch.max(outputs[0], 1)

      # statistics
      running_corrects += torch.sum(preds == labels.data)
      epoch_size += args.batch_size #inputs.size(0)

      if batch_idx % display_steps == 0:
          print('batch=', batch_idx+1, '/', len(test_loader), 'accuracy_reconstructed=', running_corrects.item() / epoch_size, 'sparsity_reconstructed=', (1-(net_ones_reconstruct/net_sizes)) )

      #for clean data  bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
      rescaled_data = backup_inputs.to(device)  #transform_to_256(data)
      #rescaled_data = data

      
      outputs_clean = model(rescaled_data)
      _, preds_clean = torch.max(outputs_clean[0], 1)

      running_corrects_clean += torch.sum(preds_clean == labels.data)

      net_ones_orig += outputs_clean[1]


      img_grid = torchvision.utils.make_grid(rescaled_data[:5])

      vutils.save_image(img_grid.detach(),
                        './real_sample_images/real_samples_idx_%03d.png' % (batch_idx),
                        normalize=True)


      
      if batch_idx % display_steps == 0:
          print('batch=', batch_idx+1, '/', len(test_loader), 'accuracy_clean=', running_corrects_clean.item() / epoch_size, 'sparsity_clean=', (1-(net_ones_orig/net_sizes))  )
          print(' ')

          
      del labels, outputs, preds, data, data_hat,z_star #, my_data_hat
     

  test_acc_reconstructed = running_corrects.item() / epoch_size
  test_acc_clean = running_corrects_clean.item() / epoch_size



  print(' ')
  print('**********************final stats************************ ')
  print('rec_iter=', rec_iter, 'rec_rr=', rec_rr, 'accu_clean=', test_acc_clean, 'accu_reconstructed', test_acc_reconstructed)
  print ('sparsity_clean=', (1-(net_ones_orig/net_sizes)), 'sparsity_reconstructed=', (1-(net_ones_reconstruct/net_sizes)) )
  print(' ')




  #*******************saving z_sets********************************
  DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
  TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)
  
  z_sets_name = 'zets_dataset'
  z_sets_name = os.path.join(z_sets_name, '_lr_', str(args.lr), '_', TIME_NOW, '.pt')
  z_sets_name = z_sets_name.replace('/',  '')
  z_sets_name = './z_sets/'+z_sets_name
  
  z_sets_array_dataset = Z_Stars_Dataset(z_sets_array)
  
  if not os.path.isdir("z_sets"):
      os.mkdir("z_sets")
    
  torch.save(z_sets_array_dataset, z_sets_name)#'./z_sets/zets_dataset.pt')


  #*******************saving z_star********************************
  z_stars_array_dataset = Z_Stars_Dataset(z_stars_array)

  if not os.path.isdir("z_stars"):
      os.mkdir("z_stars")
    
  torch.save(z_stars_array_dataset, './z_stars/zstar_dataset.pt')

  
  return 



def adjust_lr(optimizer, cur_lr, decay_rate = 0.1, global_step = 1, rec_iter = 200):
    
    lr = cur_lr * decay_rate ** (global_step / int(math.ceil(rec_iter * 0.032))) # earlier it was 0.8 but 0.8 * 200 = 0.032 * 5000
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr




    
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/cifar10/cifar10_dataset', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=20, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--rec_iter', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--rec_rr', type=int, default=20, help='number of epochs to train for')

    parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.0002')


    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manualSeed', type=int,  default=13, help='manual seed')

    #parser.add_argument('--gen_path', default='../../pytorch_dcgan/1_train/weight_dir/netG_epoch_99.pth', help='path to generator')
    #parser.add_argument('--disc_path', default='../../pytorch_dcgan/1_train/weight_dir/netD_epoch_99.pth', help='path to generator')

    parser.add_argument('--gen_path', default='../1_train/ngf_128_ndf_128_nz_512_32_32_weight_dir/netG_epoch_99.pth', help='path to generator')
    parser.add_argument('--disc_path', default='../1_train/ngf_128_ndf_128_nz_512_32_32_weight_dir/netD_epoch_99.pth', help='path to generator')
  

    parser.add_argument('--cifar10_weights', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/cifar10/2_copy_weight_256_256/alexnet_cifar10_fc_one_out_458624.pkl', help='path to generator')


   
    #parser.add_argument('--cifar10_weights', default='/home/ehsanaf/Ehsan/eg_pytorch/TrojAI/sparsity_attack/cifar10/1_train_64_64/models/alexnet_cifar_12.pkl', help='path to generator')

    parser.add_argument('--cont_z_geneartion', action='store_true', help="Use z_sets from previous round to start with")

    parser.add_argument('--im_index_fisrt', type=int, default=9000, help='The fisrt index of the dataset')
    parser.add_argument('--im_index_last', type=int, default=9100, help='The last index of the dataset')

    parser.add_argument('--use_loss_per_image', action='store_true', help="The loss function is computed per image not per batch")
    
    args = parser.parse_args()
    print(args)


    print('started at:', datetime.datetime.now() )


    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)



    device = "cuda" if torch.cuda.is_available() else "cpu" 

    ngpu = int(args.ngpu)
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)
    nc=3


    gen=Generator(ngpu, nz=nz, ngf=ngf).to(device)   
    gen.load_state_dict(torch.load(args.gen_path, map_location=device))

    disc=Discriminator(ngpu, ndf=ndf).to(device)   
    disc.load_state_dict(torch.load(args.disc_path, map_location=device))


    #cifar10 model
    model_cifar10 = AlexNet().to(device)
    model_cifar10.load_state_dict(torch.load(args.cifar10_weights, map_location=device))
    model_cifar10.eval()

    dataset = torch.load('../3_convert_dirt_dataset_batch_one/adv_cifar10_lr_0.4.pt', map_location=torch.device('cpu'))

    dataset_sub = torch.utils.data.Subset(dataset,  list(range(args.im_index_fisrt, args.im_index_last)))        

    dataset_loader = DataLoader(
       dataset_sub,
       batch_size=args.batch_size,
       shuffle=False
       )

   
    print('dataset_loader : {}'.format(len(dataset_loader)))   

    
    if args.cont_z_geneartion:
        z_sets_loader = torch.load('./z_sets/zets_dataset_in.pt', map_location=torch.device('cuda'))
    else:
       
        z_dataset = dset.CIFAR10(root=args.dataset, download=True, train=False, transform=transforms.ToTensor()  )

        z_dataset_sub = torch.utils.data.Subset(z_dataset,  list(range(args.im_index_fisrt, args.im_index_last)))        

       
        z_sets_loader = DataLoader(
            z_dataset_sub,
            batch_size=args.batch_size,
            shuffle=False
        )
  

    
    if not os.path.isdir('real_sample_images'):
        os.mkdir('real_sample_images')

    if not os.path.isdir('fake_sample_images'):
        os.mkdir('fake_sample_images')



    defense_on_clean_img(model_cifar10, gen, disc, dataset_loader, z_sets_loader, device, args)  #sigma, is_defense_gan = False):

    print('ended at:', datetime.datetime.now() )

 


