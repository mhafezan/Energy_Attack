"""
https://github.com/jiweibo/ImageNet
"""

import argparse
import os, sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from models.resnet import resnet18
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate

def train(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for index, (image, target) in enumerate(train_loader):
        # To measure image loading time
        data_time.update(time.time() - end)

        image =  image.to(device)
        target = target.to(device)

        # To compute output
        output = model(image)
        loss = criterion(output, target)

        # To measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # To compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # To measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if index % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(epoch, index, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5))
            
    return

def validate(val_loader, model, criterion, device, print_freq):
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for index, (image, target) in enumerate(val_loader):
        
        image =  image.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            # To compute output
            output = model(image)
            loss = criterion(output, target)

            # To measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), image.size(0))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            # To measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(index, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ImageNet Training on Resnet18')
    parser.add_argument('--dataset', default="../../Imagenet_dataset", help="The path to the train, val, and test datasets of ImageNet")
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful to restart from a checkpoint)')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='Weight decay (default: 1e-4)')
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true', help='To use pin memory')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='To use a pre-trained model')
    parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint, (default: None)')
    args = parser.parse_args()
    print(args)
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing!\n")

    best_pred = 0.0

    # Network Initialization
    if args.pretrained:
        print("=> Using a pre-trained model to train ResNet18")
        model = resnet18(pretrained=True).to(device)
    else:
        print("=> Training ResNet18 model from scratch")
        model = resnet18(pretrained=False)

    # To define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\n=> loading checkpoint '{args.resume}'\n")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_pred = checkpoint['best_pred']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"\n=> loaded checkpoint '{args.resume}' at epoch {checkpoint['epoch']}\n")
    else:
        print(f"\n=> no checkpoint found at '{args.resume}'\n")

    # Data loading
    train_loader, val_loader = data_loader(args.dataset, args.batch_size, args.workers, args.pin_memory)

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args.lr)

        # Train for one epoch
        train (train_loader, model, criterion, optimizer, epoch, device, args.print_freq)

        # To evaluate on validation set
        acc1, acc5 = validate (val_loader, model, criterion, device, args.print_freq)

        # To save checkpoint
        is_best = acc1 > best_pred
        best_pred = max(acc1, best_pred)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_pred': best_pred,
            'optimizer': optimizer.state_dict()}, is_best, 'checkpoints/resnet18_checkpoint_epoch_' + str(epoch+1) + '.pth')
        
    sys.exit(0)
    
# python3 train.py --batch-size 256 --pretrained