# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:14:08 2021

@author: Hira
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:18:16 2021

@author: Hira
"""

import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset_class import *
from labels_ganerator_CNN2 import *

lab = labels_CNN2_test

# TRANSFORM

transform = transforms.Compose(
    [transforms.ToTensor()])



def main():
    

    model=torch.load('weight/cnn2.pth')
    model.eval()
    
    CNN2_dataset_test = CNN_Dataset(data='hdf5_files_CNN2/test_set/' , labels=lab , transform=transform)
    data_loader_CNN2_test = DataLoader(CNN2_dataset_test , batch_size=1, shuffle=False)
    
    loss_fn = nn.CrossEntropyLoss()
    CNN2_history_test = {
        'test_loss' : [],
        'test_accuracy': []
        }
    
    predicted_labels_cnn2=[]
    
    mean_loss=[]
    print('TESTING')
    for i, (inputs, labels) in enumerate(data_loader_CNN2_test):
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            mean_loss.append(loss.item())
            predicted=torch.argmax(outputs,1)
            total += labels.size(0)
            correct+=(predicted==labels).sum()
            predicted_labels_cnn2=torch.argmax(outputs,1)
            
    CNN1_history_test['test_accuracy'].append((100*correct/len((data_loader_CNN1_test))))
    CNN1_history_test['test_loss'].append(sum(mean_loss)/len(mean_loss))
    
    
    
# Main
        
if __name__ == "__main__":
    main()

'''
model = model()  # Initialize model
model.load_state_dict(torch.load(PATH_TO_MODEL))  # Load pretrained parameters
model.eval()
'''