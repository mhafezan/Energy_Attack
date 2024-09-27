import os
import math
import torch
import torch.nn as nn

"""
model_urls: 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
"""

models_dir = os.path.expanduser('./weights')
model_name = 'vgg16-397923af.pth'

class VGG(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True, args=None):
        super(VGG, self).__init__()
        
        self.beta = args.beta
        
        self.features = nn.Sequential(
            
    	    nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
    	    nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
    
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    	    # nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(

    	    nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
    
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
    
            nn.Linear(4096, num_classes))
        
        if init_weights:
            self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        """
        
        x = self.features[0](x)
        x = self.features[1](x)

        x = self.features[2](x)
        x = self.features[3](x)
        x = self.features[4](x)

        x = self.features[5](x)
        x = self.features[6](x)

        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)

        x = self.features[10](x)
        x = self.features[11](x)

        x = self.features[12](x)
        x = self.features[13](x)

        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)

        x = self.features[17](x)
        x = self.features[18](x)

        x = self.features[19](x)
        x = self.features[20](x)

        x = self.features[21](x)
        x = self.features[22](x)
        x = self.features[23](x)

        x = self.features[24](x)
        x = self.features[25](x)

        x = self.features[26](x)
        x = self.features[27](x)

        x = self.features[28](x)
        x = self.features[29](x)
        x = self.features[30](x)

        x = x.view(x.size(0), -1)

        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)

        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)

        x = self.classifier[6](x)
        
        return x

def vgg16(pretrained=False, args=None, device=None, **kwargs):
    """
    If pretrained=True, it returns a pre-trained model on VGG16
    """
    if pretrained:
        kwargs['init_weights'] = False
        
    model = VGG(args=args, **kwargs)

    # The pretrained weights for ImageNet_on_VGG16 are trained without BatchNorm2d Layers
    if pretrained:
        model.load_state_dict(torch.load(os.path.join(models_dir, model_name)))

    return model.to(device)