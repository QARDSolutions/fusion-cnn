# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 16:23:30 2021

@author: Hanan
"""


# Import Libraries

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle

# Import Files

from dataset_class import *
from CNN2 import *
from labels_ganerator_CNN2 import labels_CNN2, labels_CNN2_val

# Hyperparameters

LEARNING_RATE = 2e-5
BATCH_SIZE = 5
EPOCHS = 50
lab = labels_CNN2
lab1 = labels_CNN2_val

# TRANSFORM

transform = transforms.Compose(
    [transforms.ToTensor()])

# Training

def main():
    
    model = CNN2()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ACCESSING DATASET
    
    CNN2_dataset_train = CNN_Dataset(data='hdf5_files_CNN2/train_set/' , labels=lab , transform=transform)
    CNN2_dataset_valid = CNN_Dataset(data='hdf5_files_CNN2/validation_set/' , labels=lab1 , transform=transform)
    #CNN2_dataset_test = CNN_Dataset(data='hdf5_files_CNN2/test_set/' , labels , transform=transform)
    
    # CALLING DATA LOADER

    data_loader_CNN2_train = DataLoader(CNN2_dataset_train , batch_size=32, shuffle=True)
    data_loader_CNN2_valid = DataLoader(CNN2_dataset_valid , batch_size=1, shuffle=False)
    #data_loader_CNN2_test = DataLoader(CNN1_dataset_test , batch_size=5, shuffle=False)
    
    # loss
    
    loss_fn = nn.CrossEntropyLoss()
    
    loop = 0
    
    CNN2_history = {
        'train_loss' : [],
        'val_loss' : [],
        'train_acc' : [],
        'valid_acc' : []
        }
    for epoch in range(EPOCHS):
        loop = loop+1
        print ("Epoch # " , loop)
        mean_loss = []
        mean_loss_valid = []
        correct = 0
        correct_valid=0
        
        for i, (inputs, labels) in enumerate(data_loader_CNN2_train):
            print ("Batch # " , i+1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            mean_loss.append(loss.item())
            out= torch.argmax(outputs,1)
            correct += (out==labels).sum().item()
            
            loss.backward()
            optimizer.step()
            
          
        # Accuracy
        accuracy= 100*(correct/len(CNN2_dataset_train))
        print()
        
        print("Epoch # " , loop , "Training Loss = " , sum(mean_loss)/len(mean_loss), "Training Accuracy = ", accuracy)
        print()
        CNN2_history['train_loss'].append(sum(mean_loss)/len(mean_loss))
        CNN2_history['train_acc'].append(accuracy)
        
        
        print('VALIDATING')
        
        for i, (inputs_valid, labels_valid) in enumerate(data_loader_CNN2_valid):
            outputs_valid = model(inputs_valid)
            loss_valid = loss_fn(outputs_valid, labels_valid)
            mean_loss_valid.append(loss.item())
            out_valid= torch.argmax(outputs_valid,1)
            correct_valid += (out_valid==labels_valid).sum().item()
     
         # Accuracy
        accuracy_valid= 100*(correct_valid/len(CNN2_dataset_train))
        print()
     
        print("Epoch # " , loop , "Validation Loss = " , sum(mean_loss_valid)/len(mean_loss_valid), "Validation Accuracy = " , accuracy_valid)
        CNN2_history['val_loss'].append(sum(mean_loss_valid)/len(mean_loss_valid))
        CNN2_history['valid_acc'].append(accuracy_valid)
        
    PATH = "cnn2.pth"
    torch.save(model.state_dict(), PATH)
    with open('CNN2_history.pickle', 'wb') as f:
        pickle.dump(CNN2_history,f)
        
# Main
        
if __name__ == "__main__":
    main()