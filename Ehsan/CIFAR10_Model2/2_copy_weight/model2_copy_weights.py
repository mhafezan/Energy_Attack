import torch
import sys
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
from models.source_model2 import Model2_stat
from models.model2 import Model2

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
       
    parser = argparse.ArgumentParser(description="Customized Model2 Network with CIFAR10 Dataset")
    parser.add_argument('--weights_src', default='../1_train_model2/weights/model2_epoch_6_0.001_64_70.69.pth', help="The path to the trained weights")
    parser.add_argument('--weights_dest', default='./model2_cifar10_fc_one_out.pkl', help="The path to store copied weights")
    parser.add_argument('--batch_size', default=64, type=int, help="Batch size")
    parser.add_argument('--dataset', default="../cifar10_dataset", help="The path to the MNIST dataset")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--im_index_first', default=0, type=int, help="The first index of the dataset")
    parser.add_argument('--im_index_last', default=10000, type=int, help="The last index of the dataset")
    args = parser.parse_args()
    print(f"\n{args}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_src = Model2_stat().to(device)
    model_dest = Model2(args=args).to(device)

    if torch.cuda.is_available():
        model_src.load_state_dict(torch.load(args.weights_src))
    else:
        model_src.load_state_dict(torch.load(args.weights_src, map_location=torch.device('cpu')))

    copy_weights(model_src, model_dest)

    model_dest.model2_set_weights_one()

    torch.save(model_dest.state_dict(), './model2_cifar10_fc_one_out.pkl')
    
    # Load the CIFAR10 dataset
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])])

    test_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # To test both models using the assigned weights
    test (model_src, test_loader, device)
    test (model_dest, test_loader, device)
    
    sys.exit()

# python3 model2_copy_weights.py