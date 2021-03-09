#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:58:35 2021

@author: danish
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
from torch import nn

# Import Files
from dataset_class import CNN_Dataset
from denseunet import UNet, dice_multiclass
from labels_ganerator_CNN2 import labels_CNN2

# Hyperparameters
LEARNING_RATE = 2e-5
EPOCHS = 50
lab = labels_CNN2
lab1 = labels_CNN2

# TRANSFORM
transform = transforms.Compose(
    [transforms.ToTensor()])

# Training
def main():
    # ACCESSING DATASET
    CNN1_dataset_train = CNN_Dataset(data='hdf5_files_CNN1/train_set/', labels=lab,
                                transform=transform, return_input='image')
    CNN1_dataset_valid = CNN_Dataset(data='hdf5_files_CNN1/validation_set/', 
                                labels=lab1, transform=transform,
                                return_input='image')
    CNN2_dataset_train = CNN_Dataset(data='hdf5_files_CNN2/train_set/', 
                                labels=lab, transform=transform, 
                                return_input='segment')
    CNN2_dataset_valid = CNN_Dataset(data='hdf5_files_CNN2/validation_set/' , 
                                labels=lab1, transform=transform, 
                                return_input='segment')
    #CNN2_dataset_test = CNN_Dataset(data='hdf5_files_CNN2/test_set/' , labels , transform=transform)

    # CALLING DATA LOADER
    data_loader_CNN1_train = DataLoader(CNN1_dataset_train , batch_size=32, shuffle=True)
    data_loader_CNN1_valid = DataLoader(CNN1_dataset_valid , batch_size=1, shuffle=False)
    data_loader_CNN2_train = DataLoader(CNN2_dataset_train , batch_size=32, shuffle=True)
    data_loader_CNN2_valid = DataLoader(CNN2_dataset_valid , batch_size=1, shuffle=False)
    #data_loader_CNN2_test = DataLoader(CNN1_dataset_test , batch_size=5, shuffle=False)

    model = UNet(in_channels=1, out_channels=1, n_blocks=4, start_filters=32,
                activation='relu', normalization='batch', conv_mode='same',
                dim=2)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loop = 0

    CNNF_history = {
    'train_loss' : [],
    'val_loss' : []}
    # Training 
    # loss
    for epoch in range(EPOCHS):
        loop = loop+1
        print ("Epoch # " , loop)
        mean_loss = []
        mean_loss_valid = []
        correct = 0
        correct_valid=0
        
        i=0
        for (inputs1, _),(segmented_images, _) in zip(data_loader_CNN1_train, data_loader_CNN2_train):
            print ("Batch # " + str(i+1))
            optimizer.zero_grad()
            outputs = model(inputs1)
            loss = loss_fn(outputs, segmented_images)
            mean_loss.append(loss.item())
            out= torch.argmax(outputs,1)
            correct += (out==labels_CNN1).sum().item()
            loss.backward()
            optimizer.step()
            i+=1
        print("Epoch # " , loop , "Training Loss = " , sum(mean_loss)/len(mean_loss))
        print()
        CNNF_history['train_loss'].append(sum(mean_loss)/len(mean_loss))
        
        for (inputs1_valid, _),(segmented_images_valid, _) in zip(data_loader_CNN1_valid, data_loader_CNN2_valid):
            outputs_valid = model(inputs1_valid)
            loss_valid = loss_fn(outputs_valid, segmented_images_valid)
            mean_loss_valid.append(loss.item())
        print()
        print("Epoch # " , loop , "Validation Loss = " , sum(mean_loss_valid)/len(mean_loss_valid))
        CNNF_history['val_loss'].append(sum(mean_loss_valid)/len(mean_loss_valid))
        
        print('VALIDATING')
        
    PATH = "cnnfUNET.pth"
    torch.save(model.state_dict(), PATH)
    with open('CNNF_historyUNET.pickle', 'wb') as f:
        pickle.dump(CNNF_history,f)
        

        
# Main
        
if __name__ == "__main__":
    main()