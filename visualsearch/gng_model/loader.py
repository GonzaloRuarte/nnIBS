import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import StratifiedGroupKFold
import go_no_go
import random
import numpy as np
import pandas as pd
import copy

class PosteriorDataset(Dataset):
    def __init__(self,x,y,fixation_nums,image_ids):
        sequence_start = np.where(fixation_nums == 1)[0]
        sequence_end = np.append(sequence_start[1:]-1,[fixation_nums.shape[0]-1])
        sequence_intervals = np.stack((sequence_start,sequence_end),axis=1)
        scanpath_ids = np.empty(shape=0)
        for index in range(0,sequence_intervals.shape[0]):
            scanpath_size = sequence_intervals[index][1] - sequence_intervals[index][0] + 1
            scanpath_ids = np.append(scanpath_ids,np.full(scanpath_size,index))
        self.x = torch.tensor(x,dtype=torch.float32,device="cuda")
        self.y = torch.tensor(y,dtype=torch.float32,device="cuda")
        self.fixation_nums = torch.tensor(fixation_nums,dtype=torch.float32,device="cuda")
        self.length = self.x.shape[0]
        self.image_ids = torch.tensor(image_ids,dtype=torch.float32,device="cuda")
        self.scanpath_ids = torch.tensor(scanpath_ids,dtype=torch.float32,device="cuda")
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx],self.fixation_nums[idx],self.scanpath_ids[idx]
    def __len__(self):
        return self.length
    def get_labels(self):
        return self.y
    def get_groups(self):
        return self.image_ids

class DoublePosteriorDataset(Dataset):
    def __init__(self,x,y,fixation_nums,image_ids):
        #obtengo los índices de comienzo y fin de scanpath
        sequence_start = np.where(fixation_nums == 1)[0]
        sequence_end = np.append(sequence_start[1:]-1,[fixation_nums.shape[0]-1])
        sequence_intervals = np.stack((sequence_start,sequence_end))
        scanpath_ids = np.empty(shape=0)
        for index in range(0,sequence_intervals.shape[1]):
            scanpath_size = sequence_intervals[1][index] - sequence_intervals[0][index] + 1
            scanpath_ids = np.append(scanpath_ids,np.full(scanpath_size,index))
        #filtro los de tamaño 1
        non_size_1_intervals = np.where(sequence_intervals[0,:] != sequence_intervals[1,:])[0]
        sequence_intervals = np.stack((sequence_intervals[0,:][non_size_1_intervals],sequence_intervals[1,:][non_size_1_intervals]),axis=1) #agrupo de a pares (principio, fin)

        #obtengo los intervalos completos y después tomo de a pares consecutivos
        #por ej si tengo un scanpath de 4 fijaciones que arranca en el índice i obtengo los pares (i,i+1), (i+1,i+2), (i+2,i+3)  
        consecutive_elements = lambda x: [[x[i], x[i + 1]] for i in range(len(x) - 1)]
        get_paired_sequences = lambda x: np.array(consecutive_elements(np.linspace(x[0],x[-1],1+x[-1]-x[0],dtype=np.int32)))
        
        full_intervals = np.concatenate(list(map(get_paired_sequences,sequence_intervals)))        
        self.intervals_indexes = full_intervals
        self.x = torch.tensor(x,dtype=torch.float32,device="cuda")
        self.y = torch.tensor(y,dtype=torch.float32,device="cuda")
        self.fixation_nums = torch.tensor(fixation_nums,dtype=torch.float32,device="cuda")
        
        self.length = self.intervals_indexes.shape[0]  
        self.image_ids = torch.tensor(image_ids[self.intervals_indexes.T[0]],dtype=torch.float32,device="cuda")
        self.scanpath_ids = torch.tensor(scanpath_ids,dtype=torch.float32,device="cuda")
    def __getitem__(self,idx):
        interval = self.intervals_indexes[idx]
        return self.x[interval[0]:interval[1]+1],self.y[interval[1]],self.fixation_nums[interval[1]],self.scanpath_ids[interval[1]]

    def __len__(self):
        return self.length

    def get_labels(self):

        return self.y[self.intervals_indexes.T[1]]

    def get_groups(self):
        return self.image_ids

class SeqDataset(Dataset):
    def __init__(self,x,y,fixation_nums,image_ids):
        #obtengo los índices de comienzo y fin de scanpath
        sequence_start = np.where(fixation_nums == 1)[0]
        sequence_end = np.append(sequence_start[1:]-1,[fixation_nums.shape[0]-1])
        sequence_intervals = np.stack((sequence_start,sequence_end))
        #filtro los de tamaño 1
        non_size_1_intervals = np.where(sequence_intervals[0,:] != sequence_intervals[1,:])[0]
        sequence_intervals = np.stack((sequence_intervals[0,:][non_size_1_intervals],sequence_intervals[1,:][non_size_1_intervals]),axis=1) #agrupo de a pares (principio, fin)

        #obtengo los intervalos completos y después tomo de a pares consecutivos
        #por ej si tengo un scanpath de 4 fijaciones que arranca en el índice i obtengo los pares (i,i+1), (i+1,i+2), (i+2,i+3)  
        consecutive_elements = lambda x: [[x[i], x[i + 1]] for i in range(len(x) - 1)]
        get_paired_sequences = lambda x: np.array(consecutive_elements(np.linspace(x[0],x[-1],1+x[-1]-x[0],dtype=np.int32)))
        
        full_intervals = np.concatenate(list(map(get_paired_sequences,sequence_intervals)))

        self.x = torch.tensor(x,dtype=torch.float32,device="cuda")
        self.y = torch.tensor(y,dtype=torch.float32,device="cuda")
        self.fixation_nums = torch.tensor(fixation_nums,dtype=torch.float32,device="cuda")
        self.intervals_indexes = full_intervals
        self.length = self.intervals_indexes.shape[0]  
        self.image_ids = torch.tensor(image_ids,dtype=torch.float32,device="cuda")
        self.scanpath_ids = torch.tensor(np.arange(len(sequence_start)),dtype=torch.float32,device="cuda")
    def __getitem__(self,idx):
        interval = self.intervals_indexes[idx]

        return self.x[interval[0]:interval[1]+1],self.y[interval[0]:interval[1]+1],self.fixation_nums[interval[0]:interval[1]+1]

    def __len__(self):
        return self.length

    def get_labels(self):

        return self.y[self.intervals_indexes.T[1]]

    def get_groups(self):
        return self.image_ids[self.intervals_indexes.T[0]]

class ModelLoader():
    def __init__(self,num_classes=1,learning_rate=0.001,epochs=1,batch_size=128,loss_fn=nn.BCEWithLogitsLoss(),optim=torch.optim.SGD,scheduler= ReduceLROnPlateau,model=go_no_go.Net,dataset=DoublePosteriorDataset):

        self.model_class = model
        self.model = model(num_classes=num_classes)
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optim_module = optim        
        self.model = self.model.to("cuda")
        self.scheduler=scheduler
        self.optim_func= self.optim_module(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.learning_rate, momentum=0.9)
        self.scheduler_func=self.scheduler(self.optim_func, 'min')
        self.dataset = dataset
    def balanced_weights(self,y_data):
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=(y_data==0.).sum()/(y_data.sum()))

    
    def load(self,model_dict_path):
        self.model.load_state_dict(torch.load(model_dict_path))

    def fit(self,posteriors,labels,fixation_nums,images_ids):    
        seed = 321
        random.seed(seed)
        # Set fixed random number seed
        torch.manual_seed(seed)
        
        trainset = self.dataset(posteriors,labels,fixation_nums,images_ids)
        del posteriors,labels,fixation_nums
        #DataLoader
        trainloader = DataLoader(trainset,self.batch_size,shuffle=False)                        
        self.model.train()
        self.balanced_weights(labels)
        #forward loop
        for epoch in range(self.epochs):
                correct, total, true_positives, true_negatives, positives, negatives = 0, 0, 0, 0, 0, 0
                # Print epoch
                print(f'Starting epoch {epoch+1}')

                # Set current loss value
                current_loss = 0.0
                self.model.train()
                # Iterate over the DataLoader for training data
                for j,(x_train,y_train,fixation_num_train,scanpath_id_train) in enumerate(trainloader):

                    # Zero the gradients
                    self.optim_func.zero_grad()
                    
                    # Perform forward pass
                    outputs = self.model(x_train,fixation_num_train)
                    predictions = (torch.sigmoid(outputs) >= 0.5)                    
                    positives += (y_train ==1).sum().item()
                    negatives += (y_train ==0).sum().item()
                    
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
                total = positives + negatives
                correct = true_positives + true_negatives
                print('TPR after epoch %d: %.3f %%' % (epoch+1,100.0 * true_positives / positives))
                print('TNR after epoch %d: %.3f %%' % (epoch+1,100.0 * true_negatives / negatives))
                print('Accuracy after epoch %d: %.3f %%' % (epoch+1,100.0 * correct / total))
                self.scheduler_func.step(loss.item())
        torch.save(self.model.state_dict(), "GNG_model_dict.pth")

    def continue_search(self,posterior,num_fixation):
                
        self.model.eval()
        
        with torch.no_grad():
            prediction = self.model(torch.tensor(np.array([posterior]),dtype=torch.float32,device="cuda"),torch.tensor(np.array([num_fixation]),dtype=torch.float32,device="cuda"))
        print(torch.sigmoid(prediction).item())
        return (torch.sigmoid(prediction) >= 0.5).item()


    def reset_weights(self):
        
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()

    def predict(self,posteriors,labels,fixation_nums,images_ids):
        testset = self.dataset(posteriors,labels,fixation_nums,images_ids)
        del posteriors,labels,fixation_nums
        self.load("gng-fold-1.pth")


        #DataLoader
        testloader = DataLoader(testset,self.batch_size,shuffle=False)
        self.model.eval()

        correct, total, true_positives, true_negatives, positives, negatives = 0, 0, 0, 0, 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for j,(x_test,y_test,fixation_num_test,scanpath_id_test) in enumerate(testloader):

                # Generate outputs
                outputs = self.model(x_test,fixation_num_test)
                
                # Set total and correct
                predictions = (torch.sigmoid(outputs) >= 0.5)
                total += y_test.size(0)

                positives += (y_test ==1).sum().item()
                negatives += (y_test ==0).sum().item()

                correct += (predictions.flatten() == y_test).sum().item()
                true_positives += torch.logical_and(predictions.flatten(),y_test).sum().item()
                true_negatives += torch.logical_and(torch.logical_not(predictions.flatten()),torch.logical_not(y_test)).sum().item()
                
                del predictions, x_test, y_test, fixation_num_test, outputs,scanpath_id_test

            # Print accuracy
            print('Accuracy in testing set: %.3f %%' % (100.0 * correct / total))
            print('TPR in testing set: %.3f %%' % (100.0 * true_positives / positives))
            if negatives > 0:
                print('TNR in testing set: %.3f %%' % (100.0 * true_negatives / negatives))






    
    def cross_val(self,posteriors,labels,fixation_nums,images_ids,k_folds=5):
        seed = 321
        training_info = pd.DataFrame(columns=["n_fold","n_epoch","acc","tpr","tnr","loss","train","valid"])
        random.seed(seed)
        # For fold results
        results = {}
        tprs = {}
        tnrs = {}
        # Set fixed random number seed
        torch.manual_seed(seed)
        
        trainset = self.dataset(posteriors,labels,fixation_nums,images_ids)
        # del posteriors,labels,fixation_nums
        # Define the K-fold Cross Validator
        kfold = StratifiedGroupKFold(n_splits=k_folds, shuffle=True,random_state=seed)
            
        # Start print
        print('--------------------------------')
        
        # K-fold Cross Validation model evaluation
        
        fold = 0

        for train_index, test_index in kfold.split(np.zeros(trainset.length), trainset.get_labels().cpu().detach().numpy(),groups=trainset.get_groups().cpu().detach().numpy()):

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
            self.model = self.model_class(num_classes=self.num_classes)   
            self.model = self.model.to("cuda")

            self.model.reset_tl_params()
            self.balanced_weights(trainset.get_labels())
            self.optim_func= self.optim_module(filter(lambda p: p.requires_grad, self.model.parameters()),lr=0.001, momentum=0.1)
            self.scheduler_func=self.scheduler(self.optim_func, 'min')
            
            # Run the training loop for defined number of epochs
            training_outputs= np.empty(shape=0)
            validation_outputs= np.empty(shape=0)
            fixation_num_training= np.empty(shape=0)
            labels_training= np.empty(shape=0)
            scanpath_ids_training= np.empty(shape=0)
            fixation_num_validation= np.empty(shape=0)
            labels_validation= np.empty(shape=0)
            scanpath_ids_validation= np.empty(shape=0)
            max_tpr = 0.0
            for epoch in range(self.epochs):
                correct, total, true_positives, true_negatives, positives, negatives = 0, 0, 0, 0, 0, 0
                # Print epoch
                print(f'Starting epoch {epoch+1}')

                # Set current loss value
                current_loss = 0.0
                current_loss_val = 0.0
                amount_minibatches = 0
                self.model.train()
                # Iterate over the DataLoader for training data
                for j,(x_train,y_train,fixation_num_train,scanpath_id_train) in enumerate(trainloader):

                    # Zero the gradients
                    self.optim_func.zero_grad()
                    
                    # Perform forward pass
                    outputs = self.model(x_train,fixation_num_train)
                    predictions = (torch.sigmoid(outputs) >= 0.5)
                    
                    
                    positives += (y_train ==1).sum().item()
                    negatives += (y_train ==0).sum().item()
                    
                    true_positives += torch.logical_and(predictions.flatten(),y_train).sum().item()
                    true_negatives += torch.logical_and(torch.logical_not(predictions.flatten()),torch.logical_not(y_train)).sum().item()
                    # Compute loss
                    loss = self.loss_fn(outputs, y_train.reshape(-1,1))
                    if epoch +1 == self.epochs:
                        fixation_num_training = np.append(fixation_num_training,fixation_num_train.cpu().detach().numpy())
                        labels_training = np.append(labels_training,y_train.cpu().detach().numpy())
                        training_outputs = np.append(training_outputs,outputs.cpu().detach().numpy())
                        scanpath_ids_training = np.append(scanpath_ids_training,scanpath_id_train.cpu().detach().numpy())

                    # Perform backward pass
                    loss.backward()
                    amount_minibatches +=1
                    # Perform optimization
                    self.optim_func.step()
                    current_loss += loss.item()
                    del  outputs, x_train, y_train, fixation_num_train, predictions
                    
                total = positives + negatives
                current_loss /= amount_minibatches
                correct = true_positives + true_negatives
                training_info = pd.concat([training_info, pd.DataFrame({"n_fold": fold,"n_epoch": epoch+1,"acc": 100.0 * correct / total,"tpr": 100.0 * true_positives / positives,"tnr": 100.0 * true_negatives / negatives,"loss": current_loss,"train": 1,"valid": 0},index=[0])],ignore_index=True)
                print('TPR after epoch %d: %.3f %%' % (epoch+1,100.0 * true_positives / positives))
                print('TNR after epoch %d: %.3f %%' % (epoch+1,100.0 * true_negatives / negatives))
                print('Accuracy after epoch %d: %.3f %%' % (epoch+1,100.0 * correct / total))
                print('Average loss for minibatches after epoch %d: %.3f' %(epoch+1, current_loss))
                self.scheduler_func.step(loss.item())
                self.model.eval()
                correct, total, true_positives, true_negatives, positives, negatives = 0, 0, 0, 0, 0, 0
                amount_minibatches = 0
                with torch.no_grad():
                    # Iterate over the test data and generate predictions
                    for j,(x_test,y_test,fixation_num_test,scanpath_id_test) in enumerate(testloader):
                        # Generate outputs
                        outputs = self.model(x_test,fixation_num_test)
                        loss = self.loss_fn(outputs, y_test.reshape(-1,1))
                        current_loss_val += loss.item()
                        amount_minibatches +=1
                        # Set total and correct
                        predictions = (torch.sigmoid(outputs) >= 0.5)
                        if epoch +1 == self.epochs:
                            fixation_num_validation = np.append(fixation_num_validation,fixation_num_test.cpu().detach().numpy())
                            labels_validation = np.append(labels_validation,y_test.cpu().detach().numpy())
                            validation_outputs = np.append(validation_outputs,outputs.cpu().detach().numpy())
                            scanpath_ids_validation = np.append(scanpath_ids_validation,scanpath_id_test.cpu().detach().numpy())
                        positives += (y_test ==1).sum().item()
                        negatives += (y_test ==0).sum().item()

                        true_positives += torch.logical_and(predictions.flatten(),y_test).sum().item()
                        true_negatives += torch.logical_and(torch.logical_not(predictions.flatten()),torch.logical_not(y_test)).sum().item()
                        
                        del predictions, x_test,y_test, fixation_num_test,outputs
                # Print accuracy
                current_loss_val /= amount_minibatches
                total = positives + negatives
                correct = true_positives + true_negatives
                training_info = pd.concat([training_info, pd.DataFrame({"n_fold": fold,"n_epoch": epoch+1,"acc": 100.0 * correct / total,"tpr": 100.0 * true_positives / positives,"tnr": 100.0 * true_negatives / negatives,"loss": current_loss_val,"train": 0,"valid": 1},index=[0])],ignore_index=True)
                print('TPR in validation set after epoch %d: %.3f %%' % (epoch+1, 100.0 * true_positives / positives))
                print('TNR in validation set after epoch %d: %.3f %%' % (epoch+1, 100.0 * true_negatives / negatives))
                print('Accuracy in validation set after epoch %d: %.3f %%' % (epoch+1, 100.0 * correct / total))
                print('Average loss for minibatches in validation after epoch after epoch %d: %.3f' %(epoch+1, current_loss_val))
                if true_positives/positives > max_tpr:
                    max_tpr = true_positives/positives
                    early_stopping = copy.deepcopy(self.model.state_dict())
            # Process is complete.
            print('Training process has finished. Saving trained model.')
            
            # Saving the model
            save_path = f'./gng-fold-{fold}.pth'
            torch.save(self.model.state_dict(), save_path)
            torch.save(early_stopping,f'./early_stopping-fold-{fold}.pth')
            np.savez_compressed(f"./{self.model.__class__.__name__}-outputs-{fold}.npz",outputs_training=training_outputs,labels_training=labels_training,fixations_training=fixation_num_training,scanpath_ids_training=scanpath_ids_training,
            outputs_validation=validation_outputs,labels_validation=labels_validation,fixations_validation=fixation_num_validation,scanpath_ids_validation=scanpath_ids_validation)


            print('--------------------------------')
            results[fold] = 100.0 * (correct / total)
            tprs[fold] = 100.0 * true_positives / positives
            tnrs[fold] = 100.0 * true_negatives / negatives
            
        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')

        for key, value in results.items():
            print(f'Fold {key} accuracy: {value} %')
        training_info.to_csv('training_info_2channels.csv')

        print(f'Average Accuracy: {sum(results.values())/len(results.items())} %')
        print(f'Average TPR: {sum(tprs.values())/len(tprs.items())} %')
        print(f'Average TNR: {sum(tnrs.values())/len(tnrs.items())} %')
