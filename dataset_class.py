# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:24:58 2021

@author: Hanan
"""

# Importing Libraries

from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
import cv2
import torch

# DATASET CLASS

class CNN_Dataset(Dataset):
    
    # CONSTRUCTOR
    
    def __init__(self, data,labels,transform, return_input='image'):
        self.data = data
        self.all_images = os.listdir(data)
        self.labels = labels
        self.transform = transform
        self.return_input = return_input

    # LENGTH FUNCTION
                
    def __len__(self):
        return len(self.all_images)
    
    # GET ITEM FUNCTION
    
    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_name = os.path.join(self.data+image_name)
        data = h5py.File(image_name,'r')
        if self.return_input=='image':
            image = data['image'][:]
        elif self.return_input=='segment':
            image = data['label'][:]
        image = np.expand_dims(image,axis=2)
        image = cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
        image = self.transform(image)
        image=image.type(torch.FloatTensor)
        label = self.labels[idx]
        
        return image , label

'''


# ACCESSING DATASET

CNN1_dataset_train = CNN_Dataset(data='hdf5_files_CNN1/train_set/' , labels = labels)
CNN1_dataset_test = CNN_Dataset(data='hdf5_files_CNN1/test_set/' , labels = labels)

CNN2_dataset_train = CNN_Dataset(data='hdf5_files_CNN2/train_set/')
CNN2_dataset_test = CNN_Dataset(data='hdf5_files_CNN2/test_set/')

# CALLING DATA LOADER

data_loader_CNN1_train = DataLoader(CNN1_dataset_train , batch_size=5, shuffle=False)
data_loader_CNN1_test = DataLoader(CNN1_dataset_test , batch_size=5, shuffle=False)

data_loader_CNN2_train = DataLoader(CNN2_dataset_train , batch_size=5, shuffle=False)
data_loader_CNN2_test = DataLoader(CNN2_dataset_test , batch_size=5, shuffle=False)

# CHECKING BATCHES

for i_batch, sample_batched in enumerate(data_loader_CNN1_train):
    if i_batch == 10 :
        break
    else :
        print("Batch #",i_batch, " ----> ", sample_batched['image'].size(), " ----> ", sample_batched['images_ids'])

print()
print()  
    
for i_batch, sample_batched in enumerate(data_loader_CNN2):
    if i_batch == 10 :
        break
    else :
        print("Batch #",i_batch, " ----> ", sample_batched['image'].size(), " ----> ", sample_batched['images_ids'])
        '''