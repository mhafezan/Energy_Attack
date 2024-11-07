import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import random
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from models.model3 import Model3

def train(model, train_loader, test_loader, optimizer, scheduler, device, args):
    
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

            # Loss over the batch progress (Printing the loss every x batches)
            if (index + 1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{index+1}/{len(train_loader)}], Loss: {loss.item():.4f}', '\n')

        # Average loss over the epoch progress (Printing the loss after each epoch execution)
        print(f'Epoch [{epoch+1}/{args.epochs}] ********** Average Loss: {running_loss/len(train_loader):.4f} **********', '\n')

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
        print(f'Epoch [{epoch+1}/{args.epochs}] ********** Test Accuracy: {accuracy:.2f}% *********', '\n')

        if args.optimizer == 'sgd':
            scheduler.step()
        else:
            scheduler.step(accuracy)
            current_lr = scheduler.optimizer.param_groups[0]['lr']
            print(f"Learning Rate after Epoch [{epoch+1}]: {current_lr:.6f}\n")

        # Save the model parameters after each epoch
        if accuracy >= 40:
            model_save_path = f'{args.weights}/model2_epoch_{epoch+1}_{args.lr}_{args.batch_size}_{accuracy:.2f}.pth'
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

def initialize_weights(model):
    # Initializes the weights of the given model. Uses Kaiming Initialization for CONV layers and Xavier Initialization for FC layers
    
    for layer in model.modules():
        """if isinstance(layer, nn.Conv2d):
            # Kaiming Initialization for convolutional layers
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)"""
        if isinstance(layer, nn.Linear):
            # Xavier Initialization for fully connected layers
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR100 Training on substitute Model3')
    parser.add_argument('--dataset', default="../cifar100_dataset", help="The path to the CIFAR100 dataset")
    parser.add_argument('--weights', default="./weights", help="The path to store model parameters")
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='Number of total epochs to train')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='Mini-batch size')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--optimizer', default="sgd", help="Optimizer option is either adam or sgd")
    parser.add_argument('--print_freq', default=100, type=int, help="To print loss and accuracy statistics every X mini-batch")
    parser.add_argument('--manualSeed', type=int, default=15, help='manual seed')
    parser.add_argument('--phase', default="train", help="To specify train or test phase")
    parser.add_argument('--model_path', default=None, help="The path to the pre-trained model for testing purpose")
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
    model = Model3().to(device)

    # Initialize weights
    model.apply(initialize_weights)
    
    # CIFAR100 dataset and dataloader declaration
    CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    CIFAR100_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    TRANSFORM = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    train_dataset = torchvision.datasets.CIFAR100(root=args.dataset, train=True, download=True, transform=TRANSFORM)
    test_dataset = torchvision.datasets.CIFAR100(root=args.dataset, train=False, download=True, transform=TRANSFORM)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    if args.phase == "train":
        train(model, train_loader, test_loader, optimizer, scheduler, device, args)
    else:
        test(model, test_loader, device, args.model_path)

    sys.exit(0)

# python3 train_test_model3.py --phase train --epochs 100 --lr 0.01 --batch-size 256 --print_freq 20 --optimizer sgd
# python3 train_test_model3.py --phase test --batch-size 128 --model_path ./weights/model2_epoch_22_0.01_256_42.76.pth
