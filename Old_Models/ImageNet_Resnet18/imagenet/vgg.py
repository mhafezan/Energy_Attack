import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class VGG(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        # we break down the x = self.features(x) into minor layers 
        x = self.features[0](x)
        x = self.features[1](x)
        L1_zeros, L1_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)
        L2_zeros, L2_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[5](x)
        x = self.features[6](x)
        L3_zeros, L3_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        L4_zeros, L4_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[10](x)
        x = self.features[11](x)
        L5_zeros, L5_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[12](x)
        x = self.features[13](x)
        L6_zeros, L6_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        L7_zeros, L7_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[17](x)
        x = self.features[18](x)
        L8_zeros, L8_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[19](x)
        x = self.features[20](x)
        L9_zeros, L9_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)
        L10_zeros, L10_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[24](x)
        x = self.features[25](x)
        L11_zeros, L11_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[26](x)
        x = self.features[27](x)
        L12_zeros, L12_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.features[28](x)
        x = self.features[29](x)
        x = self.features[30](x)
        L13_zeros, L13_size = sparsity(x) # mohammad_sparsity_calculation

        x = torch.flatten(x, 1)

        x = self.classifier[0](x)
        x = self.classifier[1](x)
        L14_zeros, L14_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[2](x)
        L15_zeros, L15_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        L16_zeros, L16_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[5](x)
        L17_zeros, L17_size = sparsity(x) # mohammad_sparsity_calculation
        x = self.classifier[6](x)
        L18_zeros, L18_size = sparsity(x) # mohammad_sparsity_calculation

        return x, L1_zeros, L1_size, L2_zeros, L2_size, L3_zeros, L3_size, L4_zeros, L4_size, L5_zeros, L5_size, L6_zeros, L6_size, L7_zeros, L7_size,\
                  L8_zeros, L8_size, L9_zeros, L9_size, L10_zeros, L10_size, L11_zeros, L11_size, L12_zeros, L12_size, L13_zeros, L13_size, L14_zeros, L14_size,\
                  L15_zeros, L15_size, L16_zeros, L16_size, L17_zeros, L17_size, L18_zeros, L18_size
    
# mohammad: To compute number of zeroes and activations in a tensor
def sparsity(y):
    zeros = torch.count_nonzero(torch.eq(y, 0)).item()
    activation_count = y.numel()
    return zeros, activation_count

def vgg16(pretrained=False, model_root=None, **kwargs):
    model = VGG(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model
