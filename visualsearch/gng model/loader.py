import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import nn

from sklearn.model_selection import KFold

from go_no_go import Net

class dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32,device="cuda")
        self.y = torch.tensor(y,dtype=torch.float32,device="cuda")
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]  
    def __len__(self):
        return self.length
        

class ModelLoader():
    def __init__(self,num_classes=1,learning_rate=0.01,epochs=700,batch_size=32,loss_fn=nn.BCEWithLogitsLoss(),optim=torch.optim.SGD):

        self.model = Net(num_classes=num_classes)
        self.model.to("cuda")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optim = optim(self.model.parameters(),lr=self.learning_rate)

    def load(self,model_dict_path):
        self.model.load_state_dict(torch.load(model_dict_path))

    def fit(self,x,y):    
        # Model , Optimizer, Loss


        trainset = dataset(x,y)
        del x,y
        #DataLoader
        trainloader = DataLoader(trainset,self.batch_size,shuffle=False)
        self.model.train()
        
        #forward loop
        for i in range(self.epochs):
            for j,(x_train,y_train) in enumerate(trainloader):

                #calculate output
                output = self.model(x_train)
                #calculate loss
                loss = self.loss_fn(output,y_train.reshape(-1,1))

   
                #backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                del loss,output
        torch.save(self.model.state_dict(), "GNG_model_dict.pkl")

    def continue_search(self,posterior):        
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(torch.tensor(posterior,dtype=torch.float32))
            argmax_prediction = torch.argmax(prediction).item()
        return argmax_prediction


    def reset_weights(self):
        '''
            Try resetting model weights to avoid
            weight leakage.
        '''
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    
    
    def cross_val(self,x,y,k_folds=5):
    
        # For fold results
        results = {}
        
        # Set fixed random number seed
        torch.manual_seed(42)

        trainset = dataset(x,y)
        del x,y
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)
            
        # Start print
        print('--------------------------------')

        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(trainset)):
            
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')
            
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)
            
            # Define data loaders for training and testing data in this fold
            trainloader = DataLoader(
                            trainset, 
                            self.batch_size, sampler=train_subsampler)
            testloader = DataLoader(
                            trainset,
                            self.batch_size, sampler=test_subsampler)
            
            # Init the neural network

            self.reset_weights()
            
            # Run the training loop for defined number of epochs
            for epoch in range(self.epochs):

                # Print epoch
                print(f'Starting epoch {epoch+1}')

                # Set current loss value
                current_loss = 0.0

                # Iterate over the DataLoader for training data
                for j,(x_train,y_train) in enumerate(trainloader):

                    # Zero the gradients
                    self.optim.zero_grad()
                    
                    # Perform forward pass
                    outputs = self.model(x_train)
                    
                    # Compute loss
                    loss = self.loss_fn(outputs, y_train.reshape(-1,1))
                    
                    # Perform backward pass
                    loss.backward()
                    
                    # Perform optimization
                    self.optim.step()
                    
                    # Print statistics
                    current_loss += loss.item()
                    if j % 500 == 499:
                        print('Loss after mini-batch %5d: %.3f' %
                            (j + 1, current_loss / 500))
                        current_loss = 0.0
                    del loss, outputs, x_train, y_train   
            # Process is complete.
            print('Training process has finished. Saving trained model.')

            # Print about testing
            print('Starting testing')
            
            # Saving the model
            save_path = f'./gng-fold-{fold}.pth'
            torch.save(self.model.state_dict(), save_path)

            # Evaluationfor this fold
            correct, total = 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for i, data in enumerate(testloader, 0):

                    # Get inputs
                    inputs, targets = data

                    # Generate outputs
                    outputs = self.model(inputs)

                    # Set total and correct
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
            
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sum = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sum += value
        print(f'Average: {sum/len(results.items())} %')