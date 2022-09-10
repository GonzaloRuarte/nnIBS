import torch
from torch import nn
from torchvision import models
from os import path
from torchvision._internally_replaced_utils import load_state_dict_from_url
class Net(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes = 1000, **kwargs)
        #black and white images

        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #pretrained resnet-152 by default
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth")
        #to make it work in grayscale images
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.load_state_dict(state_dict) 
        #Transfer Learning                   
        for param in self.parameters():
            param.requires_grad = False
        self.conv2 = nn.Conv2d(512, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.avgpool2 = nn.AvgPool2d((1,1))
        self.conv3 = nn.Conv2d(64, 32, kernel_size=7, stride=2, padding=3, bias=False)
        
        #self.reduction = nn.Linear(512 * models.resnet.Bottleneck.expansion,64)
        self.fc = nn.Linear(129, num_classes,device="cuda")
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
        
    def forward(self, x, fixation_num):

        x = nn.functional.interpolate(x,size=(224,224))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)                
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgpool2(x) 
        x = torch.flatten(x,1)

        x = torch.cat((x,fixation_num[:,None]),1)
        x = self.fc(x)

        return x
    def reset_tl_params(self):
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc.reset_parameters()
