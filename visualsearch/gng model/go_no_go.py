import torch
from torch import nn
from torchvision import models

class Net(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.Bottleneck, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        #black and white images

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

