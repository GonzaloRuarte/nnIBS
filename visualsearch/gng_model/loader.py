import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from numpy import expand_dims
from go_no_go import Net
import random
import numpy as np

class dataset(Dataset):
    def __init__(self,x,y,fixation_nums):
        self.x = torch.tensor(x,dtype=torch.float32,device="cuda")
        self.y = torch.tensor(y,dtype=torch.float32,device="cuda")
        self.fixation_nums = torch.tensor(fixation_nums,dtype=torch.float32,device="cuda")
        self.length = self.x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx],self.fixation_nums[idx]
    def __len__(self):
        return self.length
        

class ModelLoader():
    def __init__(self,num_classes=1,learning_rate=0.001,epochs=50,batch_size=128,loss_fn=nn.BCEWithLogitsLoss(),optim=torch.optim.SGD,scheduler= ReduceLROnPlateau):

        self.model = Net(num_classes=num_classes)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optim_module = optim        
        self.model = self.model.to("cuda")
        self.scheduler=scheduler
        self.optim_func= self.optim_module(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.learning_rate, momentum=0.1)
        self.scheduler_func=self.scheduler(self.optim_func, 'min')
    def balanced_weights(self,y_data):
        y_data = torch.tensor(y_data,dtype=torch.float32,device="cuda")
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=(y_data==0.).sum()/(y_data.sum()))
        del y_data 
    
    def load(self,model_dict_path):
        self.model.load_state_dict(torch.load(model_dict_path))

    def fit(self,posteriors,labels,fixation_nums):    
        
        trainset = dataset(posteriors,labels,fixation_nums)
        del posteriors,labels,fixation_nums
        #DataLoader
        trainloader = DataLoader(trainset,self.batch_size,shuffle=False)
        self.model.train()
        self.balanced_weights(labels)
        #forward loop
        for i in range(self.epochs):
            for j,(x_train,y_train,fixation_num_train) in enumerate(trainloader):

                #calculate output
                output = self.model(x_train,fixation_num_train)
                #calculate loss
                loss = self.loss_fn(output,y_train.reshape(-1,1))

   
                #backprop
                self.optim_func.zero_grad()
                loss.backward()
                self.optim_func.step()
                del output
            self.scheduler_func.step(loss.item())
        torch.save(self.model.state_dict(), "GNG_model_dict.pth")

    def continue_search(self,posterior,num_fixation):
                
        self.model.eval()

        with torch.no_grad():
            prediction = self.model(torch.tensor(expand_dims(posterior, axis=(0, 1)),dtype=torch.float32,device="cuda"),torch.tensor(num_fixation,dtype=torch.float32,device="cuda"))

        return (torch.sigmoid(prediction) >= 0.5).item()


    def reset_weights(self):
        
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    
    
    def cross_val(self,posteriors,labels,fixation_nums,k_folds=5):
        seed = 321

        random.seed(seed)
        # For fold results
        results = {}
        tprs = {}
        tnrs = {}
        # Set fixed random number seed
        torch.manual_seed(seed)
        
        trainset = dataset(posteriors,labels,fixation_nums)
        # del posteriors,labels,fixation_nums
        # Define the K-fold Cross Validator
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True,random_state=seed)
            
        # Start print
        print('--------------------------------')
        
        # K-fold Cross Validation model evaluation
        kfold.get_n_splits(posteriors, labels) # arma los folds a partir de los datos
        fold = 0
        for train_index, test_index in kfold.split(posteriors, labels):
            
            # Print
            print(f'FOLD {fold}')

            fold+=1
            print('--------------------------------')
            
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = SubsetRandomSampler(train_index)
            test_subsampler = SubsetRandomSampler(test_index)
            
            # Define data loaders for training and testing data in this fold
            trainloader = DataLoader(
                            trainset, 
                            self.batch_size, sampler=train_subsampler)
            testloader = DataLoader(
                            trainset,
                            self.batch_size, sampler=test_subsampler)
            
            # Init the neural network, only the last layers are trained
            self.model = Net(num_classes=self.num_classes)   
            self.model = self.model.to("cuda")
            self.model.train()
            self.model.reset_tl_params()
            self.balanced_weights(labels)
            self.optim_func= self.optim_module(filter(lambda p: p.requires_grad, self.model.parameters()),lr=0.001, momentum=0.1)
            self.scheduler_func=self.scheduler(self.optim_func, 'min')
            
            # Run the training loop for defined number of epochs
            total_outputs= np.empty(shape=test_index.size)
            for epoch in range(self.epochs):
                correct, total, true_positives, true_negatives, positives, negatives = 0, 0, 0, 0, 0, 0
                # Print epoch
                print(f'Starting epoch {epoch+1}')

                # Set current loss value
                current_loss = 0.0
                
                # Iterate over the DataLoader for training data
                for j,(x_train,y_train,fixation_num_train) in enumerate(trainloader):

                    # Zero the gradients
                    self.optim_func.zero_grad()
                    
                    # Perform forward pass
                    outputs = self.model(x_train,fixation_num_train)
                    predictions = (torch.sigmoid(outputs) >= 0.5)
                    
                    total += y_train.size(0)
                    positives += (y_train ==1).sum().item()
                    negatives += (y_train ==0).sum().item()

                    correct += (predictions.flatten() == y_train).sum().item()
                    true_positives += torch.logical_and(predictions.flatten(),y_train).sum().item()
                    true_negatives += torch.logical_and(torch.logical_not(predictions.flatten()),torch.logical_not(y_train)).sum().item()
                    # Compute loss
                    loss = self.loss_fn(outputs, y_train.reshape(-1,1))
                    


                    # Perform backward pass
                    loss.backward()
                    
                    # Perform optimization
                    self.optim_func.step()

                    # Print statistics
                    current_loss += loss.item()
                    if j % 500 == 499:
                        print('Loss after mini-batch %5d: %.3f' %
                            (j + 1, current_loss / 500))

                        current_loss = 0.0
                    del  outputs, x_train, y_train, fixation_num_train, predictions
                print('TPR after epoch %d: %.3f %%' % (epoch+1,100.0 * true_positives / positives))
                print('TNR after epoch %d: %.3f %%' % (epoch+1,100.0 * true_negatives / negatives))
                print('Accuracy after epoch %d: %.3f %%' % (epoch+1,100.0 * correct / total))
                self.scheduler_func.step(loss.item())
            # Process is complete.
            print('Training process has finished. Saving trained model.')
            

            # Print about testing
            print('Starting testing')
            self.model.eval()

            # Saving the model
            save_path = f'./gng-fold-{fold}.pth'
            torch.save(self.model.state_dict(), save_path)
            
            # Evaluation for this fold
            correct, total, true_positives, true_negatives, positives, negatives = 0, 0, 0, 0, 0, 0
            with torch.no_grad():

                # Iterate over the test data and generate predictions
                for j,(x_test,y_test,fixation_num_test) in enumerate(testloader):

                    # Generate outputs
                    outputs = self.model(x_test,fixation_num_test)
                    
                    # Set total and correct
                    predictions = (torch.sigmoid(outputs) >= 0.5)
                    total_outputs = np.append(total_outputs,torch.sigmoid(outputs).cpu().detach().numpy())
                    total += y_test.size(0)

                    positives += (y_test ==1).sum().item()
                    negatives += (y_test ==0).sum().item()

                    correct += (predictions.flatten() == y_test).sum().item()
                    true_positives += torch.logical_and(predictions.flatten(),y_test).sum().item()
                    true_negatives += torch.logical_and(torch.logical_not(predictions.flatten()),torch.logical_not(y_test)).sum().item()
                    
                    del predictions, x_test, y_test, fixation_num_test, outputs

            # Print accuracy
            print('Accuracy in testing set for fold %d: %.3f %%' % (fold, 100.0 * correct / total))
            print('TPR in testing set for fold %d: %.3f %%' % (fold, 100.0 * true_positives / positives))
            print('TNR in testing set for fold %d: %.3f %%' % (fold, 100.0 * true_negatives / negatives))
            np.savez_compressed(f"./gng-outputs-{fold}.npz",outputs=total_outputs)


            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
            tprs[fold] = 100.0 * true_positives / positives
            tnrs[fold] = 100.0 * true_negatives / negatives
            
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')

        for key, value in results.items():
            print(f'Fold {key} accuracy: {value} %')


        print(f'Average Accuracy: {sum(results.values())/len(results.items())} %')
        print(f'Average TPR: {sum(tprs.values())/len(tprs.items())} %')
        print(f'Average TNR: {sum(tnrs.values())/len(tnrs.items())} %')
