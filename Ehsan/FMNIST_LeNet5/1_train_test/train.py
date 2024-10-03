import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import sys
from models.LeNet5 import LeNet5

def train(model, train_loader, test_loader, optimizer, device, args):
    
    for epoch in range(args.epochs):
        
        model.train()
        running_loss = 0.0
        
        for index, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Loss over the batch progress
            if (index+1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{index+1}/{len(train_loader)}], Loss: {loss.item():.4f}','\n')
        
        # Average loss over the epoch progress
        print(f'Epoch [{epoch+1}/{args.epochs}] completed! Average Loss: {running_loss/len(train_loader):.4f}','\n')
    
    # To test the Model after all epochs
    model.eval()
    correct = 0
    processed_images = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            processed_images += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / processed_images:.2f}%','\n')
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fashion-MNIST Training on a Custom LeNet5')
    parser.add_argument('--dataset', default="../fmnist_dataset", help="The path to the train and test datasets")
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='Number of total epochs to train')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='Mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--optimizer', default="sgd", help="Optimizer option is either adam or sgd")
    parser.add_argument('--print_freq', default=100, type=int, help="To print loss and accuracy statistics every X mini-batch")
    args = parser.parse_args()
    print(f"\n{args}\n")
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing!\n")
    
    # Model Initialization
    model = LeNet5().to(device)
    
    # Load the Fashion-MNIST dataset
    TRANSFORM = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(root=args.dataset, train=True, download=True, transform=TRANSFORM)
    test_dataset = torchvision.datasets.FashionMNIST(root=args.dataset, train=False, download=True, transform=TRANSFORM)    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # To set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, test_loader, optimizer, device, args)

    sys.exit(0)
    
# python3 train.py --epochs 10 --lr 0.005 --batch-size 32 --optimizer sgd