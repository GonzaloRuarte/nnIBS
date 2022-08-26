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

class ModelLoader():
    def __init__(self,input_shape,learning_rate=0.01,epochs=700):
        self.model = Net(input_shape=input_shape)
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

