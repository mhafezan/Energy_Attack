import argparse
import os
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from tqdm import tqdm
from models.vgg import vgg16
from PIL import Image
from torchvision import transforms
from helper import AverageMeter, accuracy

"""
The ground truth of the validation images is in ILSVRC2012_validation_ground_truth.txt,
where each line contains one ILSVRC2012_ID for one image, in the ascending alphabetical 
order of the image file names.
"""
def make_dataset(dir, labels, index_first, index_last):
    
    dataset = []
    images = sorted(os.listdir(dir)) # Sort the images by filename in ascending order
    
    # To define transformation and apply to each image
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalization])
    
    for i in range(index_first, index_last):
        
        img_path = os.path.join(dir, images[i])
        image = Image.open(img_path).convert('RGB') # Convert image to RGB format
        image_tensor = transform(image)
        
        label_tensor = torch.tensor(int(labels[i].strip()))

        item = (image_tensor, label_tensor)
        dataset.append(item)

    return dataset

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ImageNet Testing on VGG16')
    parser.add_argument('--dataset', default="../../Imagenet_dataset", help="The path to the train, val, and test datasets of ImageNet")
    parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--weights', default="/weights/vgg16-397923af.pth", help="The path to the weights")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--im_index_first', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=50000, type=int, help="The last index of the dataset")
    args = parser.parse_args()
    print(args)

    test_label_file = 'ILSVRC2012_validation_ground_truth.txt'
    
    # To get ground truth labels for test data
    f = open(test_label_file, 'r')
    ground_truth_labels = f.readlines()
    f.close()

    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n A {device} is assigned for processing! \n")    

    # Network Initialization
    if args.pretrained:
        print("=> Using pre-trained weights to test vgg16")
        model = vgg16(pretrained=True, args=args, device=device)
    else:
        print("=> Using a specified trained weights to test vgg16")
        model = vgg16(pretrained=False, args=args, device=device)
        model.load_state_dict(torch.load(args.weights, map_location=device))

    # Data loading code
    test_dir = os.path.join(args.dataset, 'ILSVRC2012_img_val_unified')
    test_dataset = make_dataset(test_dir, ground_truth_labels, args.im_index_first, args.im_index_last)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # To test on the model
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for index, (image, target) in enumerate(tqdm(test_loader)):

            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(image)

            # measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    
    exit(0)
    
# python3 test.py --batch-size 16 --pretrained --im_index_last 50
