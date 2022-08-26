import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F

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
        


class Net(nn.Module):
    def __init__(self,input_shape):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(input_shape,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,1)  
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def fit(self,x,y,learning_rate=0.01,epochs = 700):    
        # Model , Optimizer, Loss
        model = self.__init__(input_shape=x.shape[1])
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)
        loss_fn = nn.BCELoss()
        trainset = dataset(x,y)
        #DataLoader
        trainloader = DataLoader(trainset,batch_size=64,shuffle=False)
        model.train()

        #forward loop
        losses = []
        accur = []
        for i in range(epochs):
            for j,(x_train,y_train) in enumerate(trainloader):

                #calculate output
                output = model(x_train)

                #calculate loss
                loss = loss_fn(output,y_train.reshape(-1,1))

   
                #backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if i%50 == 0:
                #accuracy
                predicted = model(torch.tensor(x,dtype=torch.float32))
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
        torch.save(model.state_dict(), "binary_class_model.pkl")

    def continue_search(self,posterior):

        model = self.__init__(input_shape=posterior.shape[1])
        model.load_state_dict(torch.load("binary_class_model.pkl"))
        model.eval()

        with torch.no_grad():
            prediction = model(posterior)
            argmax_prediction = torch.argmax(prediction).item()
        return argmax_prediction

