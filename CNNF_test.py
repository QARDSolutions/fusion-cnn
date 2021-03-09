# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:41:05 2021

@author: Hira
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle

# Import Files

from dataset_class import *
from CNNF import *
from labels_ganerator_CNN2 import labels_CNN2

# TRANSFORM

transform = transforms.Compose(
    [transforms.ToTensor()])



def main():
    

    model=torch.load('weight/cnnf.pth')
    model.eval()
    
    CNN1_dataset_test = CNN_Dataset(data='hdf5_files_CNN1/test_set/' , labels=lab , transform=transform)
    data_loader_CNN1_test = DataLoader(CNN1_dataset_test , batch_size=1, shuffle=False)
    CNN2_dataset_test = CNN_Dataset(data='hdf5_files_CNN2/test_set/' , labels=lab , transform=transform)
    data_loader_CNN2_test = DataLoader(CNN2_dataset_test , batch_size=1, shuffle=False)
    
    loss_fn = nn.CrossEntropyLoss()
    CNNF_history_test = {
        'test_loss' : [],
        'test_accuracy': []
        }
    
    mean_loss=[]
    for (inputs1, labels_CNN1),(inputs2, labels_CNN2) in zip(data_loader_CNN1_test, data_loader_CNN2_test):
        outputs = model(inputs1, inputs2)
        loss = loss_fn(outputs, labels_CNN1)
        mean_loss.append(loss.item())
        predicted=torch.argmax(outputs,1)
        total += labels_CNN1.size(0)
        correct+=(predicted==labels).sum()
