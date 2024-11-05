# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Generator(nn.Module):
    def __init__(self, ngpu, nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        # A sequence of transposed convolutional layers (known as "deconvolution" layers) used to upsample feature maps
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # To output a tensor with nc (number of output channels, typically 1 for grayscale or 3 for RGB images)
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh()
        )
    
    # input is Z, going into a convolution
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

"""
LeakyReLU allows a small non-zero gradient when the input is negative.
This can help prevent issues where neurons get stuck during training,
which is a common problem with regular ReLU. The Sigmoid is applied
to the output logits of the model, converting them to probabilities.
"""
class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=64):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1) # To output a probability score indicating the likelihood of the image being real.
    
def weights_init(m):
    """
    In GAN, it is common to use normal distribution with mean 0 and std 0.02,
    because it has been shown to work well with the types of architectures commonly used in GANs,
    particularly DCGANs (Deep Convolutional GANs).
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='mnist, cifar10, cifar100, imagenet')
    parser.add_argument('--dataroot', default='../MNIST_dataset', required=True, help='Path to dataset')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=1)
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--imageSize', type=int, default=28, help='The height/width of the input image')
    parser.add_argument('--nz', type=int, default=100, help='Size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='number of base filters in the generator')
    parser.add_argument('--ndf', type=int, default=64, help='number of base filters in the discriminator')
    parser.add_argument('--niter', type=int, default=25, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 parameter for adam optimizer, default=0.5')
    parser.add_argument('--cuda', action='store_true', help='Enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    print(f"\n{args}\n")
 
    if not os.path.isdir('real_sample_images'):
        os.mkdir('real_sample_images')

    if not os.path.isdir('fake_sample_images'):
        os.mkdir('fake_sample_images')

    # To create checkpoint
    if not os.path.isdir('weight_dir'):
        os.mkdir('weight_dir')
        
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("\nRandom Seed: ", args.manualSeed, '\n')
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # To optimize the performance of operations on CUDA-enabled GPUs by selecting the best algorithms for convolutional layers
    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("\nWARNING: You have a CUDA device, so you should probably run with --cuda\n")

    # To load data (Training)
    TRANSFORM = transforms.Compose([transforms.Resize(args.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    dataset = dset.MNIST(root=args.dataroot, download=True, transform=TRANSFORM)
    nc=1
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))

    # Device Initialization    
    device = torch.device("cuda:0" if args.cuda else "cpu")
    ngpu = int(args.ngpu)
    
    nz = int(args.nz)
    ngf = int(args.ngf)
    ndf = int(args.ndf)

    # Model Initialization
    netG = Generator(ngpu).to(device)
    netG.apply(weights_init) # To apply a function to each layer/module within a neural network
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    print('\n', netG, '\n')

    netD = Discriminator(ngpu).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    print('\n', netD, '\n')

    # Binary Cross-Entropy Loss function quantifies the difference between predicted probability distribution and true binary labels
    criterion = nn.BCELoss()

    # To define a fixed noise vector    
    fixed_noise = torch.randn(args.batch_size, nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    # To setup optimizer
    # Recommended hyperparameters by Standard GAN (α = 0.0002, β1 = 0.5, β2 = 0.999)
    # Recommended hyperparameters by WGAN-GP (α = 0.0001, β1 = 0, β2 = 0.9)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    for epoch in range(args.niter):
        for index, data in enumerate(dataloader, 1):
            #####################################
            # (1) Update D network: To maximize log(D(x)) + log(1 - D(G(z))) to classify fake and real with high accuracy
            #####################################
            # Train with real
            netD.zero_grad()
            real_image = data[0].to(device)
            batch_size = real_image.size(0)
            label = torch.full((batch_size,), real_label, device=device) # Creating lables=1 for Real Images for Binary Classification

            output = netD(real_image)
            loss_real = criterion(output, label)
            loss_real.backward() # D gradients are computed based on real image
            D_x = output.mean().item()

            # Train with fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device) # A random noise is created in each training iteration
            fake_image = netG(noise)
            label.fill_(fake_label)
            output = netD(fake_image.detach()) # The fake image constructed by G is fed to D. Detach() prevents the gradient from being propagated back to the G, as we are only updating the Discriminator here.
            loss_fake = criterion(output, label)
            loss_fake.backward() # Backpropagates the D gradients for the fake images and these gradients are accumulated with the ones from the real images
            D_G_z1 = output.mean().item()
            lossD = loss_real + loss_fake # Total loss for the Discriminator
            optimizerD.step() # D parameters are updated using the Accumulated Gradients

            #####################################
            # (2) Update G network: To maximize log(D(G(z))) to emulate its output as a real (to fool the Discriminator)
            #####################################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_image) # After D parameters are updated, the generated fake_image is fed again to the D, but this time with no detach() and to update the Generator.
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step() # G parameters are updated

            if index % 1000 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, args.niter, index, len(dataloader), lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))
                
            if (index*args.batch_size) % 60000 == 0:
                vutils.save_image(real_image, './real_sample_images/real_samples_%03d.png' % (epoch), normalize=True)
                
                fake_image = netG(fixed_noise)
                vutils.save_image(fake_image.detach(), './fake_sample_images/fake_samples_epoch_%03d.png' % (epoch), normalize=True)

            
        torch.save(netG.state_dict(), 'weight_dir/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'weight_dir/netD_epoch_%d.pth' % (epoch))

# Arguments: python3 train.py --dataset mnist --dataroot ../mnist_dataset/train --imageSize 28 --batch_size 60 --cuda --ngpu 1 --manualSeed 13 --niter 100
