import argparse
from utee import misc, selector
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict

def main():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--type', default='cifar10', help='|'.join(selector.known_models))
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
    parser.add_argument('--gpu', default=None, help='index of gpus to use')
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
    parser.add_argument('--model_root', default='/home/mohammad/TrojanAI/Pretrained_Models/ImageNet_Vgg16_Resnet18/pytorch-playground/models', help='folder to save the model')
    parser.add_argument('--data_root', default='/home/mohammad/TrojanAI/Pretrained_Models/ImageNet_Vgg16_Resnet18/pytorch-playground/imagenet/data', help='folder to save the model')
    parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

    parser.add_argument('--input_size', type=int, default=224, help='input size of image')
    parser.add_argument('--param_bits', type=int, default=32, help='bit-width for parameters')
    parser.add_argument('--bn_bits', type=int, default=32, help='bit-width for running mean and std')
    parser.add_argument('--fwd_bits', type=int, default=32, help='bit-width for layer output')
    parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
    args = parser.parse_args()

    #args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu) # mohammad: uncomment this for gpu
    #args.ngpu = len(args.gpu) # mohammad: uncomment this for gpu
    misc.ensure_dir(args.logdir)
    args.model_root = misc.expand_user(args.model_root)
    args.data_root = misc.expand_user(args.data_root)
    args.input_size = 299 if 'inception' in args.type else args.input_size
    
    print("=================FLAGS==================")
    for k, v in args.__dict__.items():
        print('{}: {}'.format(k, v))
    print("========================================")

    #assert torch.cuda.is_available(), 'no cuda' # mohammad: uncomment this for gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load model and dataset fetcher
    model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)
    print(model_raw)
    args.ngpu = args.ngpu if is_imagenet else 1

    # eval model without quantization
    val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
    
    if args.type == 'resnet18':
        acc1, acc5, SR_L1, SR_L2, SR_L3, SR_L4, SR_L5, SR_L6, SR_L7, SR_L8, SR_L9, SR_L10, SR_L11, SR_L12, SR_L13, SR_L14, SR_L15, SR_L16, SR_L17, SR_L18, SR_L19,\
        SR_L20, SR_L21, SR_L22, SR_Net = misc.eval_model_resnet(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)
    elif args.type == 'vgg16':
        acc1, acc5, SR_L1, SR_L2, SR_L3, SR_L4, SR_L5, SR_L6, SR_L7, SR_L8, SR_L9, SR_L10, SR_L11, SR_L12, SR_L13, SR_L14, SR_L15, SR_L16, SR_L17, SR_L18, SR_Net\
        = misc.eval_model_vgg(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

    # print sf
    print(model_raw)
    res_str = "type={}, acc1={:.4f}, acc5={:.4f}".format(args.type, acc1, acc5)
    print(res_str)
    with open('Accuracy.txt', 'a') as f:
        f.write(res_str + '\n')
    
    # Printing Sparsity Statistics (mohammad_sparsity_calculation)
    if args.type == 'resnet18':
        print()
        print('Sparsity rate of L1 is:  %1.5f'   % (SR_L1))
        print('Sparsity rate of L2 is:  %1.5f'   % (SR_L2))
        print('Sparsity rate of L3 is:  %1.5f'   % (SR_L3))
        print('Sparsity rate of L4 is:  %1.5f'   % (SR_L4))
        print('Sparsity rate of L5 is:  %1.5f'   % (SR_L5))
        print('Sparsity rate of L6 is:  %1.5f'   % (SR_L6))
        print('Sparsity rate of L7 is:  %1.5f'   % (SR_L7))
        print('Sparsity rate of L8 is:  %1.5f'   % (SR_L8))
        print('Sparsity rate of L9 is:  %1.5f'   % (SR_L9))
        print('Sparsity rate of L10 is: %1.5f'   % (SR_L10))
        print('Sparsity rate of L11 is: %1.5f'   % (SR_L11))
        print('Sparsity rate of L12 is: %1.5f'   % (SR_L12))
        print('Sparsity rate of L13 is: %1.5f'   % (SR_L13))
        print('Sparsity rate of L14 is: %1.5f'   % (SR_L14))
        print('Sparsity rate of L15 is: %1.5f'   % (SR_L15))
        print('Sparsity rate of L16 is: %1.5f'   % (SR_L16))
        print('Sparsity rate of L17 is: %1.5f'   % (SR_L17))
        print('Sparsity rate of L18 is: %1.5f'   % (SR_L18))
        print('Sparsity rate of L19 is: %1.5f'   % (SR_L19))
        print('Sparsity rate of L20 is: %1.5f'   % (SR_L20))
        print('Sparsity rate of L21 is: %1.5f'   % (SR_L21))
        print('Sparsity rate of L22 is: %1.5f'   % (SR_L22))
        print('Sparsity rate of Net is: %1.5f'   % (SR_Net))
    elif args.type == 'vgg16':
        print()
        print('Sparsity rate of L1 is:  %1.5f'   % (SR_L1))
        print('Sparsity rate of L2 is:  %1.5f'   % (SR_L2))
        print('Sparsity rate of L3 is:  %1.5f'   % (SR_L3))
        print('Sparsity rate of L4 is:  %1.5f'   % (SR_L4))
        print('Sparsity rate of L5 is:  %1.5f'   % (SR_L5))
        print('Sparsity rate of L6 is:  %1.5f'   % (SR_L6))
        print('Sparsity rate of L7 is:  %1.5f'   % (SR_L7))
        print('Sparsity rate of L8 is:  %1.5f'   % (SR_L8))
        print('Sparsity rate of L9 is:  %1.5f'   % (SR_L9))
        print('Sparsity rate of L10 is: %1.5f'   % (SR_L10))
        print('Sparsity rate of L11 is: %1.5f'   % (SR_L11))
        print('Sparsity rate of L12 is: %1.5f'   % (SR_L12))
        print('Sparsity rate of L13 is: %1.5f'   % (SR_L13))
        print('Sparsity rate of L14 is: %1.5f'   % (SR_L14))
        print('Sparsity rate of L15 is: %1.5f'   % (SR_L15))
        print('Sparsity rate of L16 is: %1.5f'   % (SR_L16))
        print('Sparsity rate of L17 is: %1.5f'   % (SR_L17))
        print('Sparsity rate of L18 is: %1.5f'   % (SR_L18))
        print('Sparsity rate of Net is: %1.5f'   % (SR_Net))

if __name__ == '__main__':
    main()

# Inference: python3 main.py --type vgg16
# Inference: python3 main.py --type resnet18
