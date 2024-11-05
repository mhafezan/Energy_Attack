import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from models.source_vgg16 import vgg16_stat
from models.vgg16 import vgg16
from helper import AverageMeter, accuracy

def copy_weights(model_source_in, model_dest_out):

    for name1, layer1 in model_source_in.named_modules():

        if isinstance(layer1, nn.Conv2d) or isinstance(layer1, nn.BatchNorm2d) or isinstance(layer1, nn.ReLU) or isinstance(layer1, nn.MaxPool2d) or isinstance(layer1, nn.AdaptiveAvgPool2d) or isinstance(layer1, nn.Dropout2d) or isinstance(layer1, nn.Linear) or isinstance(layer1, nn.Dropout):
            
            for name2, layer2 in model_dest_out.named_modules():
    
                if name1 == name2:
                    layer2.load_state_dict(layer1.state_dict())

    return


def test (model, test_loader, device):

    model.eval()
    
    top1 = AverageMeter()
    top5 = AverageMeter()

    for index, (image, target) in enumerate(test_loader):
        
        image =  image.to(device)
        target = target.to(device)
        
        with torch.no_grad():

            output = model(image)

            acc1, acc5 = accuracy(output[0].data, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

    print()
    print('{model} : Top1 Accuracy = {top1.avg:.2f} , Top5 Accuracy {top5.avg:.2f}'.format(model=model.__class__.__name__, top1=top1, top5=top5))

    return
    

if __name__ == '__main__':
       
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="VGG16 Network with ImageNet Dataset")
    parser.add_argument('--weights_source', default='../1_train_test/weights/vgg16-397923af.pth', help="The path to the weights resulting from training phase")
    parser.add_argument('--weights_dest', default='./vgg16_imagenet_fc_one_out.pkl', help="The path to the copied weights")
    parser.add_argument('--batch_size', default=20, type=int, help="Batch size")
    parser.add_argument('--dataset', default="../../Imagenet_dataset", help="The path to the train, val, and test datasets of ImageNet")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--im_index_first', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=50000, type=int, help="The last index of the dataset")
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_source = vgg16_stat(args=args)
    model_dest = vgg16(args=args)

    model_source.to(device)
    model_dest.to(device)

    if torch.cuda.is_available():
        model_source.load_state_dict(torch.load(args.weights_source))
    else:
        model_source.load_state_dict(torch.load(args.weights_source, map_location=torch.device('cpu')))

    copy_weights(model_source, model_dest)

    model_dest.vgg16_set_weights_one()

    torch.save(model_dest.state_dict(), './vgg16_imagenet_fc_one_out.pkl')
    
    # To load test dataset
    test_dir = os.path.join(args.dataset, 'ILSVRC2012_img_val')
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_dataset = datasets.ImageFolder(test_dir, transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization]))
    test_sub_dataset = torch.utils.data.Subset(test_dataset, list(range(args.im_index_first, args.im_index_last)))
    test_loader = DataLoader(test_sub_dataset, batch_size=args.batch_size, shuffle=False)
    
    # To test both models using the assigned weights

    test (model_source, test_loader, device)

    test (model_dest, test_loader, device)

# python3 vgg16_copy_weights.py --im_index_last 10