from model import LeNet5
import argparse
import os
import sys
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# FGSM attack function to create adversarial example (perturbed image)
def fgsm_attack(image_clean, epsilon, data_grad):
    """
    data_grad is the gradient of the loss w.r.t the input image
    """
    sign_data_grad  = data_grad.sign() # Collect the element-wise sign of the data gradient
    perturbed_image = image_clean + epsilon * sign_data_grad # Creates the perturbed image by adjusting each pixel of the input image
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # Clipping perturbed_image within [0,1] range
    return perturbed_image
    
# For each sample in test set, this computes the gradient of the loss w.r.t the input data (data_grad), creates a perturbed image with fgsm_attack (perturbed_data)
# Then checks to see if the perturbed example is adversarial. To test the model accuracy, it saves and returns some successful adversarial examples to be visualized later
def FGSM_Test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    for data, target in test_loader:

        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor to True to compute gradients with respect to input data
        data.requires_grad = True

        output = model(data)
        init_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability (mohammad added [0])

        # If the initial prediction (before attack) is wrong, don't bother FGSM to generate adversarial
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output[0], target) # mohammad added [0]

        # Zero all existing gradients and Calculate gradients of model in backward
        model.zero_grad()
        loss.backward()

        # Collect 'data gradients'
        data_grad = data.grad.data

        # Call fgsm_attack function to generate perturbed data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Final prediction after applying attack
        final_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability (mohammad added [0])

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving epsilon=0 examples (attack free)
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # In the case of misprediction, It saves some adv-examples for later visualization
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for the given epsilon after adversarial attack
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# To clip the perturbed input (if needed) so as to ensure that added distortions are within the "L2 bound eps" and does not exceed the input range of [0, 1]
def clip_tensor(tensor, eps):
    tensor_norm = torch.norm(tensor, p=2)
    max_norm = eps

    # If the L2 norm is greater than the eps, we scale down the tensor by multiplying it with a factor (max_norm / tensor_norm)
    if tensor_norm > max_norm:
        clipped_tensor = tensor * (max_norm / tensor_norm)
    else:
        clipped_tensor = tensor

    return torch.clamp(clipped_tensor, 0, 1)

# For each sample in test set, this computes the gradient of the loss w.r.t the input data (data_grad) and creates a perturbed image with sparsity_attack function
def Sparsity_Attack_Test(model, device, test_loader, coeff, args):
    
    correct_after = 0
    correct_before = 0
    L2_Norms = []
    SR_Net_Before = []
    SR_Net_After  = []
    net_zeros_before_total, net_sizes_before_total = [0] * 2
    net_zeros_after_total, net_sizes_after_total = [0] * 2

    # DEBUG
    counter = 0
    max_count = 10000
    
    for data, target in tqdm(test_loader, desc='Data Progress'):

        data, target = data.to(device), target.to(device)
        i_max = args.imax
        o_max = args.omax
        c_min = 0
        c_max = 1
        eps = args.eps
        eps_iter = args.eps_iter

        # DEBUG
        counter+=1
        if counter > max_count:
            print (f" Datapoint 0 to {counter-1} has been processed\n")
            break

        output = model(data)
        Net_zeros = output[1] + output[3] + output[5] + output[7] + output[9]
        Net_sizes = output[2] + output[4] + output[6] + output[8] + output[10]
        SR_Net = Net_zeros/Net_sizes
        init_pred = output[0].max(1, keepdim=True)[1] # To get the index of the maximum log-probability for Clean Input
        if init_pred.item() == target.item():
            correct_before += 1
        SR_Net_Before.append(SR_Net) # For the purpose of plotting SR_Net for each input image (before attack)
        net_zeros_before_total += Net_zeros
        net_sizes_before_total += Net_sizes
        
        for o in range(o_max):
            x = data
            for i in range(i_max):
                # Sets requires_grad attribute of X to True to compute gradients with respect to the input data
                x = x.clone().detach().requires_grad_(True)

                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

                output = model(x)
                l_ce = F.cross_entropy(output[0], target)
                l_sparsity = output[11] # To compute SR considering Tanh function
                l_x = l_sparsity + (coeff * l_ce)
                    
                optimizer.zero_grad()
                l_x.backward()
                
                # To compute gradient of Loss function (l_x) w.r.t input x
                # Below line is used instead of g(i) = μg(i−1) + x.grad.data, because optimizer has already been based on SGD with a momentum of 0.9
                g = x.grad.data
                x_new = x - (eps_iter * (g/torch.norm(g, p=2)))
                if args.constrained:
                    x_new = clip_tensor(x_new, eps)
                x = x_new

            output = model(x_new)
            final_pred = output[0].max(1, keepdim=True)[1]
            # If the initial prediction (before attack) is not equal to final prediction (after attack)
            if init_pred.item() != final_pred.item():
                coeff = (coeff + c_max)/2
            else:
                coeff = (coeff + c_min)/2

        # Compute the L2-Norm of difference between perturbed image (x_new) and clean image (data)
        l2norm_diff = torch.norm((x_new-data), p=2)
        L2_Norms.append(l2norm_diff)

        # Re-classify the final perturbed image (i.e., x_new)
        output = model(x_new)

        # Final prediction after applying attack
        final_pred = output[0].max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct_after += 1

        # Re-compute the sparsity rate using the perturbed input
        Net_zeros = output[1] + output[3] + output[5] + output[7] + output[9]
        Net_sizes = output[2] + output[4] + output[6] + output[8] + output[10]
        SR_Net = Net_zeros/Net_sizes
        SR_Net_After.append(SR_Net) # For the purpose of plotting SR_Net for each input image (after attack)
        net_zeros_after_total += Net_zeros
        net_sizes_after_total += Net_sizes

    # Calculate overal accuracy of all test data after sparsity attack
    final_acc = correct_after/float(len(test_loader))
    first_acc = correct_before/float(len(test_loader))

    return first_acc, final_acc, SR_Net_Before, SR_Net_After, L2_Norms, (net_zeros_before_total/net_sizes_before_total), (net_zeros_after_total/net_sizes_after_total)

if __name__ == '__main__':
    
    # Initialize parser and setting the hyper parameters
    parser = argparse.ArgumentParser(description="LenNet5 Network with MNIST Dataset")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.01, type=float, help="Initial learning rate")
    parser.add_argument('--phase', default='train', help="train, test, FGSM, Sparsity-Attack")
    parser.add_argument('--weights', default=None, help="The path to the saved weights. Should be specified when testing")
    parser.add_argument('--imax', default=100, type=int, help="Maximum iterations in the inner loop of Sparsity Attack function")
    parser.add_argument('--omax', default=5, type=int, help="Maximum iterations in the outer loop of Sparsity Attack function")
    parser.add_argument('--beta', default=15, type=int, help="Beta parameter used in Tanh function")
    parser.add_argument('--eps', default=0.9, type=float, help="L2-norm bound of epsilon for clipping purturbed data")
    parser.add_argument('--eps_iter', default=0.01, type=float, help="epsilon in each inner loop iteration")
    parser.add_argument('--constrained', action='store_true', help="To avoid clipping the generated purturbed data")
    args = parser.parse_args()
    print(f"{args}\n")

    # MNIST Test dataset and dataloader declaration
    if args.phase == 'FGSM':
        test_dataset = mnist.MNIST('./test', train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    elif args.phase == 'Sparsity-Attack':
        test_dataset = mnist.MNIST(root='./test', train=False, download=True, transform=transforms.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=1)
    else:
        train_dataset = mnist.MNIST(root='./train', train=True, transform=transforms.ToTensor())
        test_dataset = mnist.MNIST(root='./test', train=False, transform=transforms.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Device Initialization
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize the network
    model = LeNet5(args.beta).to(device)
    print(f"\n{model}\n")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    loss_fn = CrossEntropyLoss()
    
    if args.phase == 'train':
        EPOCHS = args.epochs
        prev_acc = 0
        for epoch in range(EPOCHS):
            model.train()
            for idx, (train_x, train_label) in enumerate(train_loader):
                optimizer.zero_grad()
                train_x = train_x.to(device)
                train_label = train_label.to(device)
                predicted_y = model(train_x.float())
                loss = loss_fn(predicted_y[0], train_label.long()) # mohammad added: [0]
                loss.backward()
                optimizer.step()
    
            all_correct_num = 0
            all_sample_num = 0
            model.eval()
            for idx, (test_x, test_label) in enumerate(test_loader):
                test_x = test_x.to(device)
                test_label = test_label.to(device)
                predicted_y = model(test_x.float().detach())
                predicted_y =torch.argmax(predicted_y[0], dim=-1) # mohammad added [0]
                current_correct_num = predicted_y == test_label
                all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
                all_sample_num += current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num
            print('Accuracy in Epoch %3d: %1.4f' % (epoch+1, acc*100), flush=True)
            if not os.path.isdir("models"):
                os.mkdir("models")
            torch.save(model.state_dict(), 'models/mnist_{:.4f}.pkl'.format(acc))
            if np.abs(acc - prev_acc) < 1e-4:
                break
            prev_acc = acc
        print("Training the model has been finished")

    elif args.phase == 'test':
        if args.weights is not None:
            # Load the pretrained model
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')

        all_correct_num = 0
        all_sample_num = 0
        L1_zeros_total, L1_size_total,\
        L2_zeros_total, L2_size_total,\
        L3_zeros_total, L3_size_total,\
        L4_zeros_total, L4_size_total,\
        L5_zeros_total, L5_size_total,\
        Net_zeros_total, Net_size_total = [0] * 12 # mohammad_sparsity_calculation

        # Set the model in evaluation mode
        model.eval()

        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predicted_y, L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size, tanh = model(test_x.float()) # mohammad_sparsity_calculation
            # Sparsity calculations zone begin (mohammad_sparsity_calculation)
            L1_zeros_total += L1_zeros
            L1_size_total  += L1_size
            L2_zeros_total += L2_zeros
            L2_size_total  += L2_size
            L3_zeros_total += L3_zeros
            L3_size_total  += L3_size
            L4_zeros_total += L4_zeros
            L4_size_total  += L4_size
            L5_zeros_total += L5_zeros
            L5_size_total  += L5_size
            Net_zeros_total += (L1_zeros + L2_zeros + L3_zeros + L4_zeros + L5_zeros)
            Net_size_total  += (L1_size + L2_size + L3_size + L4_size + L5_size)
            # Sparsity calculations zone end (mohammad_sparsity_calculation)
            predicted_y =torch.argmax(predicted_y, dim=-1)
            current_correct_num = predicted_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]

        acc = all_correct_num / all_sample_num
        print()
        print('Accuracy of Testing is (percent): %1.2f' % (acc*100), flush=True)
        print("Testing the model has been finished")
        
        # Printing Sparsity Statistics (mohammad_sparsity_calculation)
        SR_L1  = (L1_zeros_total/L1_size_total)
        SR_L2  = (L2_zeros_total/L2_size_total)
        SR_L3  = (L3_zeros_total/L3_size_total)
        SR_L4  = (L4_zeros_total/L4_size_total)
        SR_L5  = (L5_zeros_total/L5_size_total)
        SR_Net = (Net_zeros_total/Net_size_total)
        print()
        print('Sparsity rate of L1 is: %1.5f'   % (SR_L1))
        print('Sparsity rate of L2 is: %1.5f'   % (SR_L2))
        print('Sparsity rate of L3 is: %1.5f'   % (SR_L3))
        print('Sparsity rate of L4 is: %1.5f'   % (SR_L4))
        print('Sparsity rate of L5 is: %1.5f'   % (SR_L5))
        print('Sparsity rate of Network: %1.5f' % (SR_Net))

    elif args.phase == 'FGSM':
        if args.weights is not None:
            # Load the pretrained model
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()

        # FGSM Initialization
        epsilons = [0, 0.007, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        # Set the model in evaluation mode
        model.eval()
        accuracies = []
        examples = []

        # Run test for each epsilon
        for eps in epsilons:
            acc, ex = FGSM_Test(model, device, test_loader, eps)
            accuracies.append(acc)
            examples.append(ex)

        plt.figure(figsize=(5,5))
        plt.plot(epsilons, accuracies, "*-")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, .35, step=0.05))
        plt.title("Accuracy vs Epsilon")
        plt.xlabel("Epsilon")
        plt.ylabel("Accuracy")
        plt.show()
    
    elif args.phase == 'Sparsity-Attack':
        if args.weights is not None:
            # Load the pretrained model
            model.load_state_dict(torch.load(args.weights, map_location=device))
        else:
            print('No weights are provided.')
            sys.exit()
        
        # Set the model in evaluation mode
        model.eval()

        coeff = 0.5
        initial_accuracy, final_accuracy, sr_net_before, sr_net_after, l2_norms, sr_net_before_total, sr_net_after_total\
            = Sparsity_Attack_Test(model, device, test_loader, coeff, args)
        
        print(f"Test accuracy excluding energy attack: {initial_accuracy}")
        print(f"Test accuracy including energy attack: {final_accuracy}")
        print(f"Sparsity rate before energy attack: {sr_net_before_total}")
        print(f"Sparsity rate after energy attack: {sr_net_after_total}")
        print(f"Sparsity reduction applying energy attack: {sr_net_before_total/(sys.float_info.epsilon if sr_net_after_total == 0 else sr_net_after_total)}")
        print(f"Average difference of L2-Norms: {sum(l2_norms) / len(l2_norms)} \n")

        """
        test_ids = [i for i in range(1, len(test_loader)+1)]
        plt.figure(figsize=(10,10))
        plt.plot(test_ids, sr_net_before, "*-", label="sr-before")
        plt.plot(test_ids, sr_net_after , "s-", label="sr-after")
        plt.plot(test_ids, l2_norms , "D-", label="l2norm-diff")
        plt.yticks(np.arange(0, 1.1, step=0.1))
        plt.xticks(np.arange(0, len(test_loader), step=1))
        plt.title ("Sparsity rate before and after Energy-Attack")
        plt.xlabel("Test Samples")
        plt.ylabel("Sparsity Rate")
        plt.show()
        """

# Training: python3 main.py --phase train
# Testing:  python3 main.py --phase test --weights models/mnist_0.9876.pkl
# FGSM:     python3 main.py --phase FGSM --weights models/mnist_0.9876.pkl
# Sparsity: python3 main.py --phase Sparsity-Attack --omax 5 --imax 100 --eps_iter 0.01 --eps 0.9 --beta 15 --lr 0.01 --constrained --weights models/mnist_0.9876.pkl