import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F
from torchvision import models

from torch.utils.data import Dataset, DataLoader
class dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
    def __len__(self):
        return self.length
        


class Net(models.ResNet):
    def __init__(self, num_classes=1000, **kwargs):
        # Start with standard resnet152 defined here

        super().__init__(block = models.resnet.BasicBlock, layers = [3, 8, 36, 3], num_classes = num_classes, **kwargs)
        #black and white images
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        # Replace AdaptiveAvgPool2d with standard AvgPool2d        
        self.avgpool = nn.AvgPool2d((7, 7))
        # Convert the original fc layer to a convolutional layer. 
        self.last_conv = torch.nn.Conv2d( in_channels = self.fc.in_features, out_channels = num_classes, kernel_size = 1)
        self.last_conv.weight.data.copy_( self.fc.weight.data.view ( *self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_ (self.fc.bias.data)	 

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

        # Notice, there is no forward pass
        # through the original fully connected layer.
        # Instead, we forward pass through the last conv layer

        x = self.last_conv(x)

        return x

class ModelLoader():
    def __init__(self,num_classes=1,learning_rate=0.01,epochs=700):
        self.model = Net(num_classes=num_classes)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def load(self,model_dict_path):
        self.model.load_state_dict(torch.load(model_dict_path))

    def fit(self,x,y):    
        # Model , Optimizer, Loss

        optimizer = torch.optim.SGD(self.model.parameters(),lr=self.learning_rate)
        loss_fn = nn.BCELoss()
        trainset = dataset(x,y)
        #DataLoader
        trainloader = DataLoader(trainset,batch_size=64,shuffle=False)
        self.model.train()

        #forward loop
        losses = []
        accur = []
        for i in range(self.epochs):
            for j,(x_train,y_train) in enumerate(trainloader):

                #calculate output
                output = self.model(x_train)

                #calculate loss
                loss = loss_fn(output,y_train.reshape(-1,1))

   
                #backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i%50 == 0:
                #accuracy
                predicted = self.model(torch.tensor(x,dtype=torch.float32))
                acc = (predicted.reshape(-1).detach().numpy().round() == y).mean() 
                losses.append(loss)
                accur.append(acc)
                print("epoch {}\tloss : {}\t accuracy : {}".format(i,loss,acc))
        plt.plot(losses)
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.plot(accur)
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('accuracy')
        torch.save(self.model.state_dict(), "GNG_model_dict.pkl")

    def continue_search(self,posterior):        
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(posterior)
            argmax_prediction = torch.argmax(prediction).item()
        return argmax_prediction

