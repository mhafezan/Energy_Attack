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
from models.model2 import Model2

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
            if (index + 1) % args.print_freq == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{index+1}/{len(train_loader)}], Loss: {loss.item():.4f}', '\n')
        
        # Adjust the learning rate
        scheduler.step()

        # Average loss over the epoch progress
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

        # Save the model parameters after each epoch
        if accuracy >= 70:
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

"""def initialize_weights(model):
    # Initializes the weights of the given model. Uses Kaiming Initialization for CONV layers and Xavier Initialization for FC layers
    
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # Kaiming Initialization for convolutional layers
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.Linear):
            # Xavier Initialization for fully connected layers
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR10 Training on substitute Model2')
    parser.add_argument('--dataset', default="../cifar10_dataset", help="The path to the CIFAR-10 dataset")
    parser.add_argument('--weights', default="./weights", help="The path to store model parameters")
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='Number of total epochs to train')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='Mini-batch size')
    parser.add_argument('--lr', default=0.002, type=float, metavar='LR', help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--optimizer', default="sgd", help="Optimizer option is either adam or sgd")
    parser.add_argument('--print_freq', default=100, type=int, help="To print loss and accuracy statistics every X mini-batch")
    parser.add_argument('--manualSeed', type=int, default=15, help='manual seed')
    parser.add_argument('--phase', default="train", help="To specify train or test phase")
    parser.add_argument('--model_path', default=None, help="The path to the pretrained model")
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
    model = Model2().to(device)

    """# Initialize weights
    model.apply(initialize_weights)"""

    # Load the CIFAR10 dataset
    TRANSFORM = transforms.Compose([
        transforms.Resize(256),  # Resize the images to 256x256 to later generate adversarial images for AlexNet
        transforms.RandomHorizontalFlip(), # Randomly flip images horizontally
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])

    train_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=TRANSFORM)
    test_dataset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=TRANSFORM)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1) # Reduce LR by a factor of 0.1 every 30 epochs

    if args.phase == "train":
        train(model, train_loader, test_loader, optimizer, device, args)
    else:
        test(model, test_loader, device, args.model_path)

    sys.exit(0)

# python3 train_test_model2.py --phase train --epochs 100 --lr 0.001 --batch-size 128 --optimizer adam
# python3 train_test_model2.py --phase test --batch-size 128 --model_path ./weights/model1_epoch_20_0.001_128_99.33.pth
