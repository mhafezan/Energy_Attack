import torch
from utee import misc
import os
from imagenet import dataset
print = misc.logger.info
from IPython import embed

known_models = [
    'mnist', 'svhn', # 28x28
    'cifar10', 'cifar100', # 32x32
    'stl10', # 96x96
    'alexnet', # 224x224
    'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', # 224x224
    'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152', # 224x224
    'squeezenet_v0', 'squeezenet_v1', #224x224
    'inception_v3', # 299x299
]

def vgg16(cuda=True, model_root=None):
    print("Building and initializing vgg16 parameters")
    from imagenet import vgg
    m = vgg.vgg16(True, model_root)
    if (cuda and torch.cuda.is_available()):
        m = m.cuda() 
    return m, dataset.get, True

def vgg16_bn(cuda=True, model_root=None):
    print("Building vgg16_bn parameters")
    from imagenet import vgg
    m = vgg.vgg16_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def vgg19(cuda=True, model_root=None):
    print("Building and initializing vgg19 parameters")
    from imagenet import vgg
    m = vgg.vgg19(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def vgg19_bn(cuda=True, model_root=None):
    print("Building vgg19_bn parameters")
    from imagenet import vgg
    m = vgg.vgg19_bn(model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet18(cuda=True, model_root=None):
    print("Building and initializing resnet-18 parameters")
    from imagenet import resnet
    m = resnet.resnet18(True, model_root)
    if (cuda and torch.cuda.is_available()):
        m = m.cuda()
    return m, dataset.get, True

def resnet34(cuda=True, model_root=None):
    print("Building and initializing resnet-34 parameters")
    from imagenet import resnet
    m = resnet.resnet34(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet50(cuda=True, model_root=None):
    print("Building and initializing resnet-50 parameters")
    from imagenet import resnet
    m = resnet.resnet50(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet101(cuda=True, model_root=None):
    print("Building and initializing resnet-101 parameters")
    from imagenet import resnet
    m = resnet.resnet101(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def resnet152(cuda=True, model_root=None):
    print("Building and initializing resnet-152 parameters")
    from imagenet import resnet
    m = resnet.resnet152(True, model_root)
    if cuda:
        m = m.cuda()
    return m, dataset.get, True

def select(model_name, **kwargs):
    assert model_name in known_models, model_name
    kwargs.setdefault('model_root', os.path.expanduser('~/.torch/models'))
    return eval('{}'.format(model_name))(**kwargs)

if __name__ == '__main__':
    m1 = vgg16()
    embed()


