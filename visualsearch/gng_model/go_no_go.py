import torch
from torch import nn
from torchvision import models
from os import path
from torchvision._internally_replaced_utils import load_state_dict_from_url

class RNNModel(nn.Module):
    def __init__(self, hidden_dim= 64, layer_dim = 3, dropout_prob= 0.2, input_dim=768, num_classes = 1):
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes,device="cuda")

    def forward(self, x, fixation_num):
        
        x = torch.flatten(x,2)
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim,device="cuda").requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out
    def reset_tl_params(self):
        self.rnn.reset_parameters()
        self.fc.reset_parameters()



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
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.avgpool2 = nn.AvgPool2d((1,1))        
        self.fc = nn.Linear(128, 32)
        self.fc2 = nn.Linear(33,num_classes)
        

    def forward(self, x, fixation_num):
        x = torch.squeeze(x)
        fixation_num = torch.squeeze(fixation_num)
        x = torch.unsqueeze(x, axis=1) #para incorporar el canal (que es uno solo en este caso)

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
        x = self.fc(x)

        x = torch.cat((x,fixation_num[:,None]),1)
        x = self.fc2(x)

        return x
    def reset_tl_params(self):
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc.reset_parameters()
        self.fc2.reset_parameters()
