import torch
from torch import nn
from torchvision import models

class Net(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.BasicBlock, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        #black and white images

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)


    # Reimplementing forward pass.

    def _forward_impl(self, x):
        # Standard forward for resnet18

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        
        x = self.fc(x)

        return x

