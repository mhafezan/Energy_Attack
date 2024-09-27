import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import datetime
import os
import sys
import argparse
import random

from torch.utils.data import DataLoader, Dataset

from models.source_vgg16 import vgg16_stat
from models.vgg16 import vgg16

class AdversarialDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class LimitedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, images_per_class=10):
        super(LimitedImageFolder, self).__init__(root, transform)
        self.images_per_class = images_per_class
        self._filter_dataset()

    def _filter_dataset(self):
        # Dictionary to hold selected images per class (subfolder)
        class_images = {class_idx: [] for class_idx in range(len(self.classes))}
        
        for img_path, class_idx in self.samples:
            if len(class_images[class_idx]) < self.images_per_class:
                class_images[class_idx].append((img_path, class_idx))

        # Flatten and update the dataset samples
        self.samples = [img for sublist in class_images.values() for img in sublist]
        self.targets = [s[1] for s in self.samples]

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]
def clip_tensor(input_tensor, eps_in, batch_size_in, input_tensor_clean, min_in, max_in):

    with torch.no_grad():
        clapped_in = input_tensor

        for i in range(batch_size_in):
            torch.clamp(clapped_in[i], min=(min_in[i]-eps_in), max=(max_in[i]+eps_in), out=clapped_in[i])

        torch.clamp(clapped_in[:,0, :,:], min=((0-0.485)/0.229), max=((1-0.485)/0.229), out=clapped_in[:,0, :,:])
        torch.clamp(clapped_in[:,1, :,:], min=((0-0.456)/0.224), max=((1-0.456)/0.224), out=clapped_in[:,1, :,:])
        torch.clamp(clapped_in[:,2, :,:], min=((0-0.406)/0.225), max=((1-0.406)/0.225), out=clapped_in[:,2, :,:])

    return clapped_in

# To convert a clean image into an advesarial image
def convert_clean_to_adversarial(model_active_in, device, x_dirty, x_clean, init_pred, c_init, args):

    c_min = 0
    c_max = 1
    coeff = torch.full((args.batch_size,), c_init, device=device)

    max_clean, _ = torch.max(x_clean.view(args.batch_size, -1), dim=1)
    min_clean, _ = torch.min(x_clean.view(args.batch_size, -1), dim=1)

    x = x_dirty.clone().detach().requires_grad_(True)

    optimizer = optim.Adam([x], lr=args.lr, amsgrad=True)

    """image_loss = nn.MSELoss()"""
    sparsity_loss = nn.MSELoss()
    classification_loss = nn.CrossEntropyLoss()
        
    for epoch in range(args.imax):
            
        # zero the parameter gradients
        optimizer.zero_grad()
            
        # forward + backward + optimize
        outputs = model_active_in(x)

        final_pred = outputs[0].max(1, keepdim=False)[1]
        
        coeff = torch.stack([torch.where(init_pred[i] == final_pred[i], (coeff[i] + c_min)/2, (coeff[i] + c_max)/2) for i in range(args.batch_size)])
           
        loss1 = torch.stack([classification_loss(outputs[0][j].unsqueeze(0), init_pred[j].unsqueeze(0)) for j in range(args.batch_size)])


        """perfectt = torch.zeros(args.batch_size, 1, requires_grad=False).to(device)
        for i in range(args.batch_size):
            perfectt[i, 0] = torch.tensor([1.], requires_grad=False)"""

        perfectt = torch.ones(args.batch_size, 1, requires_grad=False).to(device)

        for j in range(args.batch_size):
            assert(outputs[1][j].unsqueeze(0) < 9111040)
                
        loss2 = torch.stack([sparsity_loss(outputs[1][j].unsqueeze(0)/9111040, perfectt[j].unsqueeze(0)) for j in range(args.batch_size)])

        """loss3 = torch.stack([image_loss(x[j], x_clean[j]) for j in range(args.batch_size)])"""
            
        loss = (coeff * loss1) + loss2
        """loss = (coeff * loss1) + loss2 + loss3"""

        loss.backward(torch.ones_like(loss))
        
        optimizer.step()

        if args.constrained:
            x = clip_tensor(x, args.eps, args.batch_size, x_clean, min_clean, max_clean)

    return x

# To generate sparsity attack for X% of each class in test_dataset
def sparsity_attack_generation(model_active_in, model_static_in, device, test_loader_clean_in, test_loader_dirty_in, num_classes, c_init, args):

    print(f"Started at: {datetime.datetime.now()}")
    
    model_active_in.eval()
    model_static_in.eval()

    num_processed_images = 0
    correct_before = 0
    correct_after = 0
    l2_norms = []
    total_net_ones_before = 0
    total_net_sizes_before = 0
    total_net_ones_after = 0
    total_net_sizes_after = 0

    adversarial_data = []    

    """net_sizes = (3*224*224 + 64*224*224 + 64*112*112 + 128*112*112 + 128*56*56 +
                    256*56*56 + 256*56*56 + 256*28*28 + 512*28*28 + 512*28*28 +
                    512*14*14 + 512*14*14 + 512*14*14 + 25088 + 4096) * args.batch_size"""
    
    net_sizes = 9111040 * args.batch_size

    count = 0
    
    for ((data_clean, target_clean) , (data_dirty, target_dirty)) in zip (test_loader_clean, test_loader_dirty):
        inputs_clean, target_clean = data_clean.to(device), target_clean.to(device)
        inputs_dirty, target_dirty = data_dirty.to(device), target_dirty.to(device)

        # To inject the Clean Image to model to compute the accuracy and sparsity rate
        output_init = model_static_in(inputs_clean)
        init_pred = output_init[0].max(1, keepdim=False)[1]
        correct_before += (init_pred == target_clean).sum().item()
        
        net_ones = output_init[1]
        total_net_ones_before += net_ones
        total_net_sizes_before += net_sizes

        count = count + 1        
        if count % 200 == 0:
            print(f"\nCount = {count} \n")
            print(datetime.datetime.now())

        x_new = convert_clean_to_adversarial (model_active_in, device, inputs_dirty, inputs_clean, init_pred, c_init, args)

        # To store the generated adversarial (x_new) or benign data in a similar dataset with a pollution rate of 100% for each class
        if args.store_attack:
            adversarial_data.append((x_new, target_clean))

        # Compute the L2-Norm of difference (Percentage) between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-inputs_clean).view(args.batch_size, -1), p=2, dim=1)
        l2norm_diff = l2norm_diff/(x_new.view(args.batch_size, -1)).size(1)
        for l2norm in l2norm_diff: l2_norms.append(l2norm.item())

        # To inject the Adversarial Image to model to compute the accuracy and sparsity rate
        output_new = model_static_in(x_new)
        final_pred = output_new[0].max(1, keepdim=False)[1]
        correct_after += (final_pred == target_clean).sum().item()

        net_ones = output_new[1]
        total_net_ones_after += net_ones
        total_net_sizes_after += net_sizes
        
        num_processed_images += args.batch_size

        if args.save_samples:
            img_grid_clean = torchvision.utils.make_grid(inputs_clean)
            vutils.save_image(img_grid_clean.detach(),'./sample_images/clean_idx_%03d.png' % (count), normalize=True)
      
            img_grid_dirty = torchvision.utils.make_grid(x_new)
            vutils.save_image(img_grid_dirty.detach(), './sample_images/dirty_idx_%03d.png' % (count), normalize=True)
            
    # To Create a new dataset using the AdversarialDataset class
    if args.store_attack:
        adversarial_dataset = AdversarialDataset(adversarial_data)
    else:
        adversarial_dataset = 0
        
    # To calculate overal accuracy of all test data after sparsity attack
    first_acc = correct_before/float(num_processed_images)
    final_acc = correct_after/float(num_processed_images)

    print(f"\nEnded at: {datetime.datetime.now()} \n")

    return adversarial_dataset, first_acc, final_acc, l2_norms, (1-(total_net_ones_before/total_net_sizes_before)), (1-(total_net_ones_after/total_net_sizes_after))



if __name__ == '__main__':
       
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="VGG16 Network with ImageNet Dataset")
    parser.add_argument('--batch_size', default=10, type=int, help="Batch size")
    parser.add_argument('--weights', default='../2_copy_weight/vgg16_imagenet_fc_one_out.pkl', help="The path to the copied weights")
    parser.add_argument('--dataset', default="../../Imagenet_dataset", help="The path to the train, val, and test datasets of ImageNet")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--constrained', action='store_true', help="To enable clipping the generated purturbed data")
    parser.add_argument('--save_samples', action='store_true', help="To save sample adversarial images")
    parser.add_argument('--store_attack', action='store_true', help="To enable or disable algorithm to store generated adversarials")
    parser.add_argument('--im_index_first', default=0, type=int, help="The fisrt index of the dataset")
    parser.add_argument('--im_index_last', default=50000, type=int, help="The last index of the dataset")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Attack function")
    parser.add_argument('--lr', default=0.001, type=float, help="Initial learning rate")
    parser.add_argument('--cont_adv_geneartion', action='store_true', help="To read adversarial images from previous imax-loop and continuously optimize the images")
    parser.add_argument('--manualSeed', type=int,  default=15, help='manual seed')
    parser.add_argument('--images_per_class', default=50, type=int, help="Maximum image selected for each class")
    args = parser.parse_args()
    print(f"\n{args} \n")

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 50000)
    print(f"\nRandom Seed: {args.manualSeed} \n")
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    
    # Device Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{device} assigned for processing! \n")

    # To load test dataset
    test_dir = os.path.join(args.dataset, 'ILSVRC2012_img_val')
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization])
    
    """test_dataset_clean = datasets.ImageFolder(test_dir, transform)"""
    test_dataset_clean = LimitedImageFolder(test_dir, transform, images_per_class=args.images_per_class)
    """test_dataset_sub_clean = torch.utils.data.Subset(test_dataset_clean, list(range(args.im_index_first, args.im_index_last)))"""
    test_loader_clean = DataLoader(test_dataset_clean, batch_size=args.batch_size, shuffle=False, num_workers=1)
 
    if args.cont_adv_geneartion:
        test_loader_dirty = torch.load('./adversarial_dataset_vgg16_in.pt', map_location=torch.device('cuda'))
    else:
        """test_dataset_dirty = datasets.ImageFolder(test_dir, transform)"""
        test_dataset_dirty = LimitedImageFolder(test_dir, transform, images_per_class=args.images_per_class)
        """test_dataset_sub_dirty = torch.utils.data.Subset(test_dataset_dirty, list(range(args.im_index_first, args.im_index_last)))"""
        test_loader_dirty = DataLoader(test_dataset_dirty, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # Network Initialization
    model_active = vgg16(args=args).to(device)
    model_static = vgg16_stat(args=args).to(device)

    if args.save_samples:
        if not os.path.isdir('sample_images'):
            os.mkdir('sample_images')

    if args.weights is not None:
        model_active.load_state_dict(torch.load(args.weights, map_location=device))
        model_static.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("No weights are provided.")
        sys.exit()

    c_init = 0.5
    num_classes = len(test_dataset_clean.classes)
    
    adversarial_dataset, initial_accuracy, final_accuracy, l2_norms, sr_net_before_total, sr_net_after_total = sparsity_attack_generation (model_active, model_static, device, test_loader_clean, test_loader_dirty, num_classes, c_init, args)
     
    # The time we run the script
    DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
    TIME_NOW = datetime.datetime.now().strftime(DATE_FORMAT)

    image_out_name = 'adversarial_dataset_vgg16'
    image_out_name = os.path.join(image_out_name, '_lr_', str(args.lr), '_', TIME_NOW, '.pt')
    image_out_name = image_out_name.replace('/', '')

    # To save the generated adversarial dataset to disk
    if args.store_attack:
        torch.save(adversarial_dataset, './'+ image_out_name)

    print(f"\nTest accuracy excluding energy attack: {initial_accuracy}")
    print(f"Test accuracy including energy attack: {final_accuracy}")
    print(f"Sparsity rate before energy attack: {sr_net_before_total}")
    print(f"Sparsity rate after energy attack: {sr_net_after_total}")
    print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
    print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)}\n")

    f_ptr = open('./tmp_sparsity_results.txt', 'a')
    print('lr=', args.lr, file = f_ptr)
    print('acc_before = ', initial_accuracy, file = f_ptr)
    print('acc_after = ', final_accuracy, file = f_ptr)
    print('sparsity_before = ', sr_net_before_total, file = f_ptr)
    print('sparsity_after = ', sr_net_after_total, file = f_ptr)
    print(' ', file = f_ptr)

# Arguments: python3 vgg16_attack.py --imax 50 --beta 15 --eps 0.99 --batch_size 10 --images_per_class 10 --store_attack --constrained --dataset ../../Imagenet_dataset --weights ../2_copy_weight/vgg16_imagenet_fc_one_out.pkl
