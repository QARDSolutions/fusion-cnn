# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:54:16 2021

@author: Hira
"""

import torch
from CNN1 import CNN1
from CNN2 import CNN2
from CNNF import CNNF
from sklearn.metrics import confusion_matrix
from labels_ganerator_CNN1 import labels_CNN1_test as true_labels1
from labels_ganerator_CNN2 import labels_CNN2_test as true_labels2
import pickle
import matplotlib.pyplot as plt
from dataset_class import CNN_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import seaborn as sns

CLASSES = ['DCM', 'HCM', 'MINF', 'ARV', 'NOR']

def loadPickle(pklPath):
    with open(pklPath, 'rb') as  f:
        history=pickle.load(f)
    return history

def plotLoss(history:dict, title:str, epochs=50):
    epoch = range(1,epochs+1)
    plt.plot(epoch, history['train_loss'], label='train loss')
    plt.plot(epoch, history['val_loss'], label='val loss')
    plt. xlabel('Epochs')
    plt. ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(title+'.png')
    plt.show()

def plot_confusion_matrix(cm, labels, title):
    """Plot confusion matrix using heatmap.
     
    Args:
        cm (list of list or 2D Array): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
     
    """
    sns.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
    plt.title(title)
    sns.set(font_scale=1.2)
    ax = sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
    plt.savefig(title+'.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

def testModel(model, test_data_path, labels:list):
    transform = transforms.Compose([transforms.ToTensor()])
    if type(test_data_path)==str:
        #dataloader
        dataset_test = CNN_Dataset(data=test_data_path, labels=labels, transform=transform)
        data_loader = DataLoader(dataset_test, batch_size=1, shuffle=False)
        #getting predictions
        print('TESTING')
        predictions = []
        for i, (inputs, label) in tqdm(enumerate(data_loader)):
            outputs = model(inputs)
            predictions.append(torch.argmax(outputs,1).numpy()[0])
    elif type(test_data_path)==list:
        #dataloader
        dataset_test1 = CNN_Dataset(data=test_data_path[0], labels=labels, transform=transform)
        data_loader1 = DataLoader(dataset_test1, batch_size=1, shuffle=False)
        dataset_test2 = CNN_Dataset(data=test_data_path[1], labels=labels, transform=transform)
        data_loader2 = DataLoader(dataset_test2, batch_size=1, shuffle=False)
        #getting predictions
        print('TESTING')
        predictions = []
        for (inputs1, _),(inputs2, _) in tqdm(zip(data_loader1, data_loader2)):
            outputs = model(inputs1, inputs2)
            predictions.append(torch.argmax(outputs,1).numpy()[0])
    return predictions

def getConfusionMatrix(true_labels, predictions, lb_names, title):
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm, lb_names, title)
    return cm




################ CNN1 Ploting ################
#loss plot
plotLoss(loadPickle('weight/CNN1_history.pickle'), title='Loss trend for CNN1')
#Confusion matrix
model1 = CNN1(chanells=1)
model1.load_state_dict(torch.load('weight/cnn1.pth'))
predictions = testModel(model1, test_data_path='hdf5_files_CNN1/test_set/', labels=true_labels1)
cm1 = getConfusionMatrix(true_labels1, predictions, lb_names=CLASSES[0:2], 
                   title='Confusion Matrix CNN1')


################ CNN2 Ploting ################
#loss plot
plotLoss(loadPickle('weight/CNN2_history.pickle'), title='Loss trend for CNN2')
#Confusion matrix
model2 = CNN2()
model2.load_state_dict(torch.load('weight/cnn2.pth'))
predictions = testModel(model2, test_data_path='hdf5_files_CNN2/test_set/', labels=true_labels2)
cm2 = getConfusionMatrix(true_labels2, predictions, lb_names=CLASSES[0:2], 
                   title='Confusion Matrix CNN2')


################ CNNF Ploting ################
#loss plot
plotLoss(loadPickle('weight/CNNF_history.pickle'), title='Loss trend for CNNF')
#Confusion matrix
modelf = CNNF()
modelf.load_state_dict(torch.load('weight/cnnf.pth'))
predictions = testModel(modelf, test_data_path=['hdf5_files_CNN1/test_set/', 
                        'hdf5_files_CNN2/test_set/'], labels=true_labels2)
cm2 = getConfusionMatrix(true_labels2, predictions, lb_names=CLASSES[0:2], 
                   title='Confusion Matrix CNNF')

