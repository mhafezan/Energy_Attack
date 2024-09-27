import shutil
import torch

# To compute and store the average and current value
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# To compute the precision for topk
def accuracy(output, target, topk=(1,)):

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


def save_checkpoint(state, is_best, filename='checkpoints/vgg16_checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'weights/vgg16_best_model.pth')

# To Set the learning rate to the initial LR decayed by 10 every 30 epochs
def adjust_learning_rate(optimizer, epoch, init_lr):

    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr