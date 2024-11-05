import argparse
import torch
import torch.nn as nn
import os
import os.path
import numpy as np
import joblib
import tqdm
from models.vgg import vgg16

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

def get_dataset(root=None, batch_size=10, im_index_first=0, im_index_last=50000, **kwargs):

    print(f"Building IMAGENET test data loader for indices {im_index_first} to {im_index_last} out of 50000")
    ds = []

    ds.append(IMAGENET(root, batch_size, im_index_first=im_index_first, im_index_last=im_index_last, **kwargs))
    
    ds = ds[0] if len(ds) == 1 else ds
    
    return ds

class IMAGENET(object):
    def __init__(self, root, batch_size, input_size=224, im_index_first=0, im_index_last=50000, **kwargs):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)

        pkl_file = os.path.join(root, 'val{}.pkl'.format(input_size))
        self.data_dict = joblib.load(pkl_file)
        
        # Limit the data to the specified index range
        self.data_dict['data'] = self.data_dict['data'][im_index_first:im_index_last]
        self.data_dict['target'] = self.data_dict['target'][im_index_first:im_index_last]

        self.batch_size = batch_size
        self.index = 0

    @property
    def n_batch(self):
        return int(np.ceil(self.n_sample * 1.0 / self.batch_size))

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
            img = self.data_dict['data'][self.index * self.batch_size:(self.index + 1) * self.batch_size].astype('float32')
            target = self.data_dict['target'][self.index * self.batch_size:(self.index + 1) * self.batch_size]
            self.index += 1
            return img, target



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="VGG16 Network with ImageNet Dataset")
    parser.add_argument('--batch_size', type=int, default=10, help='Input batch size for test')
    parser.add_argument('--weights', default="/weights/vgg16-397923af.pth", help="The path to the customized weights")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='To use pre-trained model')
    parser.add_argument('--dataset',  default="data", help="Path to the test datasets")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--im_index_first', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=50000, type=int, help="The last index of the dataset")
    args = parser.parse_args()
    print(args)
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n A {device} assigned for processing! \n")

    test_dataset = get_dataset(root=args.dataset, batch_size=args.batch_size, im_index_first=args.im_index_first, im_index_last=args.im_index_last, input_size=224)

    # Network Initialization
    if args.pretrained:
        print("=> Using pre-trained weights to test vgg16")
        model = vgg16(pretrained=True, args=args, device=device)
    else:
        print("=> Using a specified trained weights to test vgg16")
        model = vgg16(pretrained=False, args=args, device=device)
        model.load_state_dict(torch.load(args.weights, map_location=device))

    correct = 0
    num_processed_images = 0
    
    model = ModelWrapper(model)
    model.eval()
    
    n_sample = len(test_dataset)
    
    for index, (data, target) in enumerate(tqdm.tqdm(test_dataset, total=n_sample)):
    
        num_processed_images += len(data)
        data =  torch.FloatTensor(data).to(device)
        target = torch.LongTensor(target).to(device)

        output = model(data)
        
        batch_size = output.size(0)
        preds = output.data.sort(1, descending=True)[1]
        
        idx_bs = target.expand(1, batch_size).transpose_(0, 1)

        correct += preds[:, :1].cpu().eq(idx_bs).sum()
            
        if index >= n_sample - 1:
            break

    accuracy = correct * 1.0 / num_processed_images

    accuracy_results = "Top1 Accuacy: {:.2f}".format(accuracy*100)
    print()
    print(accuracy_results)
    
    exit(0)

# Test: python3 test_with_pklfile_val.py --batch_size 1 --dataset data --im_index_last 50 --pretrained --weights weights/vgg16-397923af.pth