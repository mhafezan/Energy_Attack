import torch
import sys
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from models.source_model1 import Model1_stat
from models.model1 import Model1

def copy_weights(model_src_in, model_dest_out):

    for name1, layer1 in model_src_in.named_modules():

        if isinstance(layer1, nn.Conv2d) or isinstance(layer1, nn.BatchNorm2d) or isinstance(layer1, nn.ReLU) or isinstance(layer1, nn.MaxPool2d) or isinstance(layer1, nn.AdaptiveAvgPool2d) or isinstance(layer1, nn.AvgPool2d) or isinstance(layer1, nn.Dropout2d) or isinstance(layer1, nn.Linear) or isinstance(layer1, nn.Dropout)or isinstance(layer1, nn.Softmax):
            
            for name2, layer2 in model_dest_out.named_modules():
    
                if name1 == name2:
                    layer2.load_state_dict(layer1.state_dict())

    return

def test(model, test_loader, device):
    
    model.eval()
    
    correct = 0
    processed_images = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs[0], 1)
            processed_images += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = float(correct / processed_images) * 100
    print(f'\nModel Test Accuracy: {accuracy:.2f}%\n')

    return



if __name__ == '__main__':
       
    parser = argparse.ArgumentParser(description="Customized Model1 Network with MNIST Dataset")
    parser.add_argument('--weights_src', default='../1_train_model1/weights/model1_epoch_20_0.001_128_99.33.pth', help="The path to the trained weights")
    parser.add_argument('--weights_dest', default='./model1_mnist_fc_one_out.pkl', help="The path to store copied weights")
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size")
    parser.add_argument('--dataset', default="../mnist_dataset", help="The path to the MNIST dataset")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--im_index_first', default=0, type=int, help="The first index of the dataset")
    parser.add_argument('--im_index_last', default=10000, type=int, help="The last index of the dataset")
    args = parser.parse_args()
    print(f"\n{args}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_src = Model1_stat().to(device)
    model_dest = Model1(args=args).to(device)

    if torch.cuda.is_available():
        model_src.load_state_dict(torch.load(args.weights_src))
    else:
        model_src.load_state_dict(torch.load(args.weights_src, map_location=torch.device('cpu')))

    copy_weights(model_src, model_dest)

    model_dest.model1_active_set_weights_one()

    torch.save(model_dest.state_dict(), './model1_mnist_fc_one_out.pkl')
    
    # Load the MNIST dataset
    TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
    test_dataset = torchvision.datasets.MNIST(root=args.dataset, train=False, download=True, transform=TRANSFORM)
    test_sub_dataset = torch.utils.data.Subset(test_dataset, list(range(args.im_index_first, args.im_index_last)))
    test_loader = DataLoader(test_sub_dataset, batch_size=args.batch_size, shuffle=False)
    test_sub_dataset = torch.utils.data.Subset(test_dataset, list(range(args.im_index_first, args.im_index_last)))
    
    # To test both models using the assigned weights
    test (model_src, test_loader, device)
    test (model_dest, test_loader, device)
    
    sys.exit()

# python3 model1_copy_weights.py