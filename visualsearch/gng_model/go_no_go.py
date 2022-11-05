import torch
from torch import nn
from torchvision import models
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
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, fixation_num,image):
        
        x = torch.flatten(x,2)
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

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



class TransferNet(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes = 1000, **kwargs)
        #black and white images
        self.inplanes = 64
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #pretrained resnet-152 by default
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth")
        #to make it work in grayscale images
        conv1_weight = state_dict['conv1.weight']        
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True).repeat(1,2,1,1)
        self.load_state_dict(state_dict) 
        #Transfer Learning                   
        for param in self.parameters():
            param.requires_grad = False
        self.conv2 = nn.Conv2d(512, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.avgpool2 = nn.AvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,num_classes)
        

    def forward(self, x, fixation_num,image):
        #x = torch.squeeze(x)
        #x = torch.unsqueeze(x, axis=1) #para incorporar el canal (que es uno solo en este caso)

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
        x = self.relu(x)
        x = self.avgpool2(x)
        x = torch.flatten(x,1)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    def reset_tl_params(self):
        self.conv2.reset_parameters()
        self.fc2.reset_parameters()

class TransferNetWithImage(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes = 1000, **kwargs)
        #black and white images
        self.inplanes = 64
        self.conv1 = nn.Conv2d(2, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        #pretrained resnet-152 by default
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet18-f37072fd.pth")
        #to make it work in grayscale images
        conv1_weight = state_dict['conv1.weight']        
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True).repeat(1,2,1,1)
        self.load_state_dict(state_dict) 
        #Transfer Learning                   
        for param in self.parameters():
            param.requires_grad = False
        self.conv2 = nn.Conv2d(512, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.avgpool2 = nn.AvgPool2d((1,1))
        self.fc2 = nn.Linear(1024,num_classes)
        

    def forward(self, x, fixation_num,image):
        x = torch.squeeze(x)
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
        x = self.relu(x)
        x = self.avgpool2(x)
        x = torch.flatten(x,1)

        image = self.conv1(image)
        image = self.bn1(image)
        image = self.relu(image)
        image = self.maxpool(image)

        image = self.layer1(image)
        image = self.layer2(image)
        image = self.layer3(image)
        image = self.layer4(image)
        image = self.avgpool(image)
        image = torch.flatten(image,1)

        x = self.fc2(torch.stack((x,image)))

        return x
    def reset_tl_params(self):
        self.conv2.reset_parameters()
        self.fc2.reset_parameters()



class Net(nn.Module):
    def __init__(self, num_classes=1000, **kwargs):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn2 = nn.BatchNorm2d(128)
        self.avgpool2 = nn.AvgPool2d((1,1))        
        self.fc2 = nn.Linear(512,num_classes)
        

    def forward(self, x, fixation_num,image):
        #x = torch.squeeze(x)
        #x = torch.unsqueeze(x, axis=1) #para incorporar el canal (que es uno solo en este caso)
        x = nn.functional.interpolate(x,size=(224,224))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = self.relu(x)        

        x = self.avgpool2(x)
        x = torch.flatten(x,1)

        x = self.fc2(x)

        return x
    def reset_tl_params(self):
        pass