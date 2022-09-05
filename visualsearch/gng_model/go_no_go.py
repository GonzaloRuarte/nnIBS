import torch
from torch import nn
from torchvision import models
from os import path
from torchvision._internally_replaced_utils import load_state_dict_from_url
class Net(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.Bottleneck, layers = [3, 8, 36, 3], num_classes = 1000, **kwargs)
        #black and white images

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #pretrained resnet-152 by default
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet152-394f9c45.pth")
        #to make it work in grayscale images
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.load_state_dict(state_dict) 
        #Transfer Learning                   
        for param in self.parameters():
            param.requires_grad = False
        self.reduction = nn.Linear(512 * models.resnet.Bottleneck.expansion,64)
        self.avgpool2 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(65, num_classes,device="cuda")
        
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
        x = torch.flatten(x,1)
        x = self.reduction(x)
        x = self.avgpool2(x)
        x = torch.cat((x,fixation_num[:,None]),1)
        
        x = self.fc(x)

        return x