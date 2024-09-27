"""
https://github.com/jiweibo/ImageNet
"""
import os, sys
import tqdm
import argparse
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.vgg import vgg16
from helper import AverageMeter, accuracy

def validate(test_loader, model, device):
    
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for index, (image, target) in enumerate(tqdm.tqdm(test_loader)):
        
        image =  image.to(device)
        target = target.to(device)
        
        with torch.no_grad():
            
            # To compute output
            output = model(image)

            # To measure accuracy
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

    print()
    print('Top1 Accuracy = {top1.avg:.2f} , Top5 Accuracy {top5.avg:.2f}'.format(top1=top1, top5=top5))

    return



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ImageNet Testing on VGG16')
    parser.add_argument('--dataset', default="../../Imagenet_dataset", help="The path to the train, val, and test datasets of ImageNet")
    parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='Mini-batch size (default: 256)')
    parser.add_argument('--weights', default="/weights/vgg16-397923af.pth", help="The path to the weights")
    parser.add_argument('--workers', default=1, type=int, metavar='N', help='Number of data loading workers (default: 4)')
    parser.add_argument('--pin-memory', dest='pin_memory', action='store_true', help='To use pin memory')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='To use pre-trained model')
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--im_index_first', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=50000, type=int, help="The last index of the dataset")
    args = parser.parse_args()
    print(args)
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n A {device} assigned for processing! \n")

    # Network Initialization
    if args.pretrained:
        print("=> Using pre-trained weights to test vgg16")
        model = vgg16(pretrained=True, args=args, device=device)
    else:
        print("=> Using a specified trained weights to test vgg16")
        model = vgg16(pretrained=False, args=args, device=device)
        model.load_state_dict(torch.load(args.weights, map_location=device))

    # To load test dataset
    test_dir = os.path.join(args.dataset, 'ILSVRC2012_img_val')
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_dataset = datasets.ImageFolder(test_dir, transforms.Compose([transforms.Resize(256),
                                                                      transforms.CenterCrop(224),
                                                                      transforms.ToTensor(),
                                                                      normalization]))
    
    test_sub_dataset = torch.utils.data.Subset(test_dataset, list(range(args.im_index_first, args.im_index_last)))
    
    test_loader = torch.utils.data.DataLoader(test_sub_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=args.pin_memory)
    
    # To test the dataset
    validate (test_loader, model, device)
        
    sys.exit(0)
    
# python3 test.py --batch-size 1 --pretrained --im_index_last 50 --weights weights/vgg16-397923af.pth