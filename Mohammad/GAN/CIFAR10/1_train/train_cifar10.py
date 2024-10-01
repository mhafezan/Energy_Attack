# https://github.com/csinva/gan-vae-pretrained-pytorch/blob/master/cifar10_dcgan/dcgan.py
# https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py
# python dcgan.py --dataset cifar10 --dataroot /scratch/users/vision/yu_dl/raaz.rsk/data/cifar10 --imageSize 32 --cuda --outf out_cifar --manualSeed 13 --niter 100

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
    def __init__(self, ngpu, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
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
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
            )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc=3, ndf=64):
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
            # state size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            
        return output.view(-1, 1).squeeze(1)

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
    parser.add_argument('--dataroot', required=True, help='Path to cifar10 dataset')
    parser.add_argument('--workers', type=int, default=1, help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size; 4 is a good choice for CIFAR10 Training')
    parser.add_argument('--imageSize', type=int, default=32, help='The height or width of the input image')
    parser.add_argument('--nz', type=int, default=512, help='The size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--niter', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='The beta1 parameter for adam optimizer, default=0.5')
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
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # To load data (Training)
    """
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    """
    dataset = dset.CIFAR10(root=args.dataroot, download=True, train=True, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Device Initialization
    device = torch.device("cuda:0" if args.cuda else "cpu")
    ngpu = int(args.ngpu)

    # Network Initialization
    netG = Generator(ngpu, nz=args.nz, ngf=args.ngf).to(device)
    netG.apply(weights_init)
    if args.netG != '':
        netG.load_state_dict(torch.load(args.netG))
    print('\n', netG, '\n')

    netD = Discriminator(ngpu, ndf=args.ndf).to(device)
    netD.apply(weights_init)
    if args.netD != '':
        netD.load_state_dict(torch.load(args.netD))
    print('\n', netD, '\n')

    # Binary Cross-Entropy Loss function quantifies the difference between predicted probability distribution and true binary labels
    criterion = nn.BCELoss()

    # A fixed noise vector to later visualizing the fake image
    fixed_noise = torch.randn(args.batch_size, args.nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    # To setup optimizer
    # Recommended hyperparameters by Standard GAN (α = 0.0002, β1 = 0.5, β2 = 0.999)
    # Recommended hyperparameters by WGAN-GP (α = 0.0001, β1 = 0, β2 = 0.9)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    for epoch in range(args.niter):
        for batch_index, data in enumerate(dataloader, 0):
            #####################################
            # (1) Update D network: To maximize log(D(x)) + log(1 - D(G(z))) to classify fake and real with high accuracy
            #####################################
            # Train with real
            netD.zero_grad()
            
            real_image = data[0].to(device)
            batch_size = real_image.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_image)
            loss_real = criterion(output, label)
            loss_real.backward() # First, D gradients are computed based on real image
            D_X = output.mean().item()

            # Train with fake
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device) # A random noise is created in each training iteration
            fake_image = netG(noise)
            label.fill_(fake_label)
            
            output = netD(fake_image.detach()) # The fake image constructed by G is fed to D. Detach() prevents the gradient from being propagated back to the G, as we are only updating the Discriminator here.
            loss_fake = criterion(output, label)
            loss_fake.backward() # Second, D gradients are computed based on fake images (these gradients are accumulated with the ones from the real images)
            D_G_Before = output.mean().item() # D(G(z)) before updating the NetD parameters
            
            lossD = loss_real + loss_fake
            optimizerD.step()

            #####################################
            # (2) Update G network: To maximize log(D(G(z))) to emulate its output as a real (to fool the Discriminator)
            #####################################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake_image) # The fake_image is fed again to the D, but this time with no detach() and to update the Generator.
            lossG = criterion(output, label)
            lossG.backward()
            D_G_After = output.mean().item() # D(G(z)) after updating the NetD parameters
            optimizerG.step()

            if batch_index % 1000 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, args.niter, batch_index, len(dataloader), lossD.item(), lossG.item(), D_X, D_G_Before, D_G_After))
            if (batch_index*args.batch_size) % 50000 == 0:
                vutils.save_image(real_image, './real_sample_images/real_samples_%03d.png' % (epoch), normalize=True)
                fake_image = netG(fixed_noise)
                vutils.save_image(fake_image.detach(), './fake_sample_images/fake_samples_epoch_%03d.png' % (epoch), normalize=True)
           
        torch.save(netG.state_dict(), 'weight_dir/netG_epoch_%d.pth' % (epoch))
        torch.save(netD.state_dict(), 'weight_dir/netD_epoch_%d.pth' % (epoch))

"""        
Arguments:
python3 train_cifar10.py --dataroot ../cifar10-dataset --imageSize 32 --cuda --manualSeed 13 --niter 100 --nz 512
"""