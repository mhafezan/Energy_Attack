import torch
import argparse
from torch.utils.data import DataLoader, Dataset

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def convert_to_batch_one(args):
  
    dataset = torch.load(args.data_in, map_location=torch.device('cpu'))

    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)
  
    image_array = []

    for index, (inputs, labels, adversarial) in enumerate(dataset_loader):

        inputs = inputs.squeeze(0)
        labels = labels.squeeze(0)
                                    
        for i in range(inputs.size(0)):
            temp = inputs[i]
            image_array.append((temp, labels[i], adversarial))

    image_array_dataset = AdversarialDataset(image_array)
    
    torch.save(image_array_dataset, args.data_out)
    
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_in', default=None, help="The path to the batched dirty dataset")
    parser.add_argument('--data_out', default=None, help="The path to the non-batch dirty dataset")
    args = parser.parse_args()
    print(args)
    
    convert_to_batch_one(args)

# Arguments: python3 convert_dirty_image_to_batch_one.py --data_in ./adversarial_dataset_mnist_unconstrained_lr_0.9_bs_20.pt --data_out adversarial_dataset_mnist_unconstrained_lr_0.9_bs_1.pt