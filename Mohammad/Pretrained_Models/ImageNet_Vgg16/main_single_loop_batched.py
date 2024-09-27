import argparse
import torch
import torch.nn as nn
import os
import os.path
import sys
import numpy as np
import joblib
import tqdm
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

class ModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
    def forward(self, input):
        input.data.div_(255.)
        input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
        input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
        input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
        return self.model(input)

def get_dataset (root=None, train=False, test=True, batch_size=10, **kwargs):

    print("Building IMAGENET data loader, 50000 for train, 50000 for test")
    ds = []
    assert train is not True, 'train not supported yet'
    if train:
        ds.append(IMAGENET(root, batch_size, True, **kwargs))
    if test:
        ds.append(IMAGENET(root, batch_size, False, **kwargs))
    
    ds = ds[0] if len(ds) == 1 else ds
    return ds

class IMAGENET(object):
    def __init__(self, root, batch_size, train=False, input_size=224, **kwargs):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        self.train = train

        if train:
            pkl_file = os.path.join(root, 'train{}.pkl'.format(input_size))
        else:
            pkl_file = os.path.join(root, 'val{}.pkl'.format(input_size))
            
        self.data_dict = joblib.load(pkl_file)
        self.batch_size = batch_size
        self.index = 0

    @property
    def n_batch(self):
        return int(np.ceil(self.n_sample* 1.0 / self.batch_size))

    @property
    def n_sample(self):
        return len(self.data_dict['data'])

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.n_batch:
            self.index = 0
            raise StopIteration
        else:
            img = self.data_dict['data'][self.index*self.batch_size:(self.index+1)*self.batch_size].astype('float32')
            target = self.data_dict['target'][self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            return img, target

class VGG(nn.Module):
    def __init__(self, num_classes=1000, args=None):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        # we break down the x = self.features(x) into minor layers 
        x = self.features[0](x)
        x = self.features[1](x)
        L1_zeros, L1_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        L2_zeros, L2_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[5](x)
        x = self.features[6](x)
        L3_zeros, L3_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        L4_zeros, L4_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[10](x)
        x = self.features[11](x)
        L5_zeros, L5_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[12](x)
        x = self.features[13](x)
        L6_zeros, L6_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        L7_zeros, L7_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[17](x)
        x = self.features[18](x)
        L8_zeros, L8_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[19](x)
        x = self.features[20](x)
        L9_zeros, L9_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)
        L10_zeros, L10_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[24](x)
        x = self.features[25](x)
        L11_zeros, L11_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[26](x)
        x = self.features[27](x)
        L12_zeros, L12_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[28](x)
        x = self.features[29](x)
        x = self.features[30](x)
        L13_zeros, L13_size = sparsity(x) # mohammad_sparsity_calculation

        x = torch.flatten(x, 1)

        x = self.classifier[0](x)
        x = self.classifier[1](x)
        L14_zeros, L14_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[2](x)
        L15_zeros, L15_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        L16_zeros, L16_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[5](x)
        L17_zeros, L17_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[6](x)
        L18_zeros, L18_size = sparsity(x) # mohammad_sparsity_calculation

        return x, L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size, L6_zeros, L6_size, L7_zeros, L7_size,\
                  L8_zeros, L8_size, L9_zeros, L9_size, L10_zeros, L10_size, L11_zeros, L11_size, L12_zeros, L12_size, L13_zeros, L13_size, L14_zeros, L14_size,\
                  L15_zeros, L15_size, L16_zeros, L16_size, L17_zeros, L17_size, L18_zeros, L18_size
                  
def sparsity(y):
    zeros = torch.count_nonzero(torch.eq(y, 0)).item()
    activation_count = y.numel()
    return zeros, activation_count

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="VGG16 Network with ImageNet Dataset")
    parser.add_argument('--batch_size', type=int, default=10, help='input batch size for training and test')
    parser.add_argument('--phase', default='test', help="test, profile, sparsity-attack, and sparsity-detect")
    parser.add_argument('--method', default='sparsity-map', help="To specify profiling mechanism based on sparsity-map or sparsity-range")
    parser.add_argument('--weights', type=str, required=True, help='path to the pre-trained model')
    parser.add_argument('--dataset', help="The path to the train and test datasets")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon in each inner loop iteration")
    parser.add_argument('--constrained', action='store_true', help="To enable clipping the generated purturbed data")
    parser.add_argument('--store_attack', action='store_true', help="To lead the algorithm to store the generated adversarial")
    parser.add_argument('--power', action='store_true', help="To generate power results for CNVLUTIN architecture")
    parser.add_argument('--adversarial', action='store_true', help="To test the adversarial dataset in the test phase")
    parser.add_argument('--arch', default='cnvlutin', help="To specify the architecture running the clean/adversarial image: cnvlutin or dadiannao")
    parser.add_argument('--resume', action='store_true', help="To load the last saved checkpoint.The saving is done by defualt in adversarial attack algorithm")
    args = parser.parse_args()
    print(args)
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n A {device} is assigned for processing! \n")

    # To prepare the test dataset
    if args.phase == 'test':
        test_dataset = get_dataset(root=args.dataset, train=False, test=True, batch_size=args.batch_size, input_size=224)

    # Network Initialization
    model = VGG(args=args)
    model.to(device)
    print(model)
    
    if args.phase == 'test':
        
        if args.weights is not None:
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided, downloading from model zoo')
            model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], "models"))

        correct = 0
        num_processed_images = 0

        L1_zeros_total, L1_size_total, L2_zeros_total, L2_size_total, L3_zeros_total, L3_size_total, L4_zeros_total, L4_size_total, L5_zeros_total, L5_size_total,\
        L6_zeros_total, L6_size_total, L7_zeros_total, L7_size_total, L8_zeros_total, L8_size_total, L9_zeros_total, L9_size_total, L10_zeros_total, L10_size_total,\
        L11_zeros_total, L11_size_total, L12_zeros_total, L12_size_total, L13_zeros_total, L13_size_total, L14_zeros_total, L14_size_total,\
        L15_zeros_total, L15_size_total, L16_zeros_total, L16_size_total, L17_zeros_total, L17_size_total, L18_zeros_total, L18_size_total,\
        Net_zeros_total, Net_size_total = [0]*38
            
        model = ModelWrapper(model)
        model.eval()
        
        n_sample = len(test_dataset)
        for index, (data, target) in enumerate(tqdm.tqdm(test_dataset, total=n_sample)): # To examine original dataset

            num_processed_images += len(data)
            data =  torch.FloatTensor(data).to(device)
            target = torch.LongTensor(target).to(device)
                    
            output, L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size, L6_zeros, L6_size, L7_zeros, L7_size,\
            L8_zeros, L8_size, L9_zeros, L9_size, L10_zeros, L10_size, L11_zeros, L11_size, L12_zeros, L12_size, L13_zeros, L13_size, L14_zeros, L14_size,\
            L15_zeros, L15_size, L16_zeros, L16_size, L17_zeros, L17_size, L18_zeros, L18_size = model(data)
                    
            # Sparsity calculations
            L1_zeros_total  += L1_zeros
            L1_size_total   += L1_size
            L2_zeros_total  += L2_zeros
            L2_size_total   += L2_size
            L3_zeros_total  += L3_zeros
            L3_size_total   += L3_size
            L4_zeros_total  += L4_zeros
            L4_size_total   += L4_size
            L5_zeros_total  += L5_zeros
            L5_size_total   += L5_size
            L6_zeros_total  += L6_zeros
            L6_size_total   += L6_size
            L7_zeros_total  += L7_zeros
            L7_size_total   += L7_size
            L8_zeros_total  += L8_zeros
            L8_size_total   += L8_size
            L9_zeros_total  += L9_zeros
            L9_size_total   += L9_size
            L10_zeros_total += L10_zeros
            L10_size_total  += L10_size
            L11_zeros_total += L11_zeros
            L11_size_total  += L11_size
            L12_zeros_total += L12_zeros
            L12_size_total  += L12_size
            L13_zeros_total += L13_zeros
            L13_size_total  += L13_size
            L14_zeros_total += L14_zeros
            L14_size_total  += L14_size
            L15_zeros_total += L15_zeros
            L15_size_total  += L15_size
            L16_zeros_total += L16_zeros
            L16_size_total  += L16_size
            L17_zeros_total += L17_zeros
            L17_size_total  += L17_size
            L18_zeros_total += L18_zeros
            L18_size_total  += L18_size
            Net_zeros_total += (L1_zeros + L2_zeros + L3_zeros + L4_zeros + L5_zeros + L6_zeros + L7_zeros + L8_zeros + L9_zeros + L10_zeros +\
                                L11_zeros + L12_zeros + L13_zeros + L14_zeros + L15_zeros + L16_zeros + L17_zeros + L18_zeros)
            Net_size_total  += (L1_size + L2_size + L3_size + L4_size + L5_size + L6_size + L7_size + L8_size + L9_size +\
                                L10_size + L11_size + L12_size + L13_size + L14_size + L15_size + L16_size + L17_size + L18_size)
    
            batch_size = output.size(0)
            preds = output.data.sort(1, descending=True)[1]
            
            idx_bs = target.expand(1, batch_size).transpose_(0, 1)
    
            correct += preds[:, :1].cpu().eq(idx_bs).sum()
            
            if index >= n_sample - 1:
                break

        accuracy = correct * 1.0 / num_processed_images

        # Calculating Sparsity Statistics for each layer
        SR_L1   = (L1_zeros_total/L1_size_total)
        SR_L2   = (L2_zeros_total/L2_size_total)
        SR_L3   = (L3_zeros_total/L3_size_total)
        SR_L4   = (L4_zeros_total/L4_size_total)
        SR_L5   = (L5_zeros_total/L5_size_total)
        SR_L6   = (L6_zeros_total/L6_size_total)
        SR_L7   = (L7_zeros_total/L7_size_total)
        SR_L8   = (L8_zeros_total/L8_size_total)
        SR_L9   = (L9_zeros_total/L9_size_total)
        SR_L10  = (L10_zeros_total/L10_size_total)
        SR_L11  = (L11_zeros_total/L11_size_total)
        SR_L12  = (L12_zeros_total/L12_size_total)
        SR_L13  = (L13_zeros_total/L13_size_total)
        SR_L14  = (L14_zeros_total/L14_size_total)
        SR_L15  = (L15_zeros_total/L15_size_total)
        SR_L16  = (L16_zeros_total/L16_size_total)
        SR_L17  = (L17_zeros_total/L17_size_total)
        SR_L18  = (L18_zeros_total/L18_size_total)
        SR_Net  = (Net_zeros_total/Net_size_total)

        accuracy_results = "Top1_Accuacy={:.4f}".format(accuracy)
        print()
        print(accuracy_results)
        
        # To print Sparsity Statistics
        print()
        print('Sparsity rate of L1 is:  %1.5f'   % (SR_L1))
        print('Sparsity rate of L2 is:  %1.5f'   % (SR_L2))
        print('Sparsity rate of L3 is:  %1.5f'   % (SR_L3))
        print('Sparsity rate of L4 is:  %1.5f'   % (SR_L4))
        print('Sparsity rate of L5 is:  %1.5f'   % (SR_L5))
        print('Sparsity rate of L6 is:  %1.5f'   % (SR_L6))
        print('Sparsity rate of L7 is:  %1.5f'   % (SR_L7))
        print('Sparsity rate of L8 is:  %1.5f'   % (SR_L8))
        print('Sparsity rate of L9 is:  %1.5f'   % (SR_L9))
        print('Sparsity rate of L10 is: %1.5f'   % (SR_L10))
        print('Sparsity rate of L11 is: %1.5f'   % (SR_L11))
        print('Sparsity rate of L12 is: %1.5f'   % (SR_L12))
        print('Sparsity rate of L13 is: %1.5f'   % (SR_L13))
        print('Sparsity rate of L14 is: %1.5f'   % (SR_L14))
        print('Sparsity rate of L15 is: %1.5f'   % (SR_L15))
        print('Sparsity rate of L16 is: %1.5f'   % (SR_L16))
        print('Sparsity rate of L17 is: %1.5f'   % (SR_L17))
        print('Sparsity rate of L18 is: %1.5f'   % (SR_L18))
        print('Sparsity rate of Net is: %1.5f'   % (SR_Net))

# Test: python3 main_single_loop_batched.py --phase test --batch_size 2 --dataset data --weights models/vgg16-397923af.pth