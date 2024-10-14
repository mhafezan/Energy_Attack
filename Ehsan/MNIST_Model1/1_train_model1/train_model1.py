import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import sys
import random
from model.model1 import Model1

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
        print(f'Epoch [{epoch+1}/{args.epochs}] ********** Average Loss: {running_loss/len(train_loader):.4f} **********','\n')
    
        # Test the model after each epoch
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
        
        accuracy = float(correct / processed_images) * 100
        print(f'Epoch [{epoch+1}/{args.epochs}] ********** Test Accuracy: {accuracy:.2f}% *********','\n')
        
        # Save the model parameters after each epoch
        if accuracy >= 90:
            model_save_path = f'{args.weights}/model1_epoch_{epoch+1}_{args.lr}_{args.batch_size}_{accuracy:.2f}.pth'
            torch.save(model.state_dict(), model_save_path)
        
    return

def test(model, test_loader, device, model_path):
    
    # Load the saved model parameters
    model.load_state_dict(torch.load(model_path))
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
    
    accuracy = float(correct / processed_images) * 100
    print(f'\nModel Test Accuracy: {accuracy:.2f}%\n')

    return

# Define initialization function (using Xavier Distribution) for network weights
def weights_init(model):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MNIST Training on substitute Model1')
    parser.add_argument('--dataset', default="../mnist_dataset", help="The path to the MNIST datasets")
    parser.add_argument('--weights', default="./weights", help="The path to store model parameters")
    parser.add_argument('--epochs', default=15, type=int, metavar='N', help='Number of total epochs to train')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='Mini-batch size')
    parser.add_argument('--lr', default=0.002, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--optimizer', default="sgd", help="Optimizer option is either adam or sgd")
    parser.add_argument('--print_freq', default=100, type=int, help="To print loss and accuracy statistics every X mini-batch")
    parser.add_argument('--manualSeed', type=int,  default=15, help='manual seed')
    parser.add_argument('--phase', default="train", help="To specify train or test phase")
    parser.add_argument('--model_path', default=None, help="The path to the pre-trained model")
    args = parser.parse_args()
    print(f"\n{args}\n")

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print(f"\nRandom Seed: {args.manualSeed} \n")
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing!\n")
    
    # Model Initialization
    model = Model1().to(device)

    # Initialize weights to Xavier Uniform using weights_init() function
    model.apply(weights_init)
    
    # Load the MNIST dataset
    TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.1307], std=[0.3081])])
    
    train_dataset = torchvision.datasets.MNIST(root=args.dataset, train=True, download=True, transform=TRANSFORM)
    test_dataset = torchvision.datasets.MNIST(root=args.dataset, train=False, download=True, transform=TRANSFORM)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.phase == "train":
        train(model, train_loader, test_loader, optimizer, device, args)
    else:
        test(model, test_loader, device, args.model_path)

    sys.exit(0)

# python3 train_model1.py --phase train --epochs 50 --lr 0.002 --batch-size 128 --optimizer adam
# python3 train_model1.py --phase test --batch-size 128 --model_path ./weights/model1_epoch_x.pth
