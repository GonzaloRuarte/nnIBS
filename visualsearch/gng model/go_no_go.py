import torch
from torch import nn
from torchvision import models
from os import path
from torchvision._internally_replaced_utils import load_state_dict_from_url
class Net(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.Bottleneck, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        #black and white images

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if path.exists(path.join(".","GNG_model_dict.pth")):
            self.load_state_dict(torch.load(path.join(".","GNG_model_dict.pth")))
        else:
            #pretrained resnet-152 by default
            state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet152-394f9c45.pth")
            #to make it work in grayscale images
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.load_state_dict(state_dict)
    def forward(self, x, fixation_num):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        #x = torch.cat([torch.flatten(x, 1),torch.tensor([fixation_num])])
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x